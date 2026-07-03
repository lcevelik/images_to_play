"""Minimal **binary** FBX 7.4 (a.k.a. "FBX 2013") writer for an animated camera.

Why this exists: the old exporter wrote ASCII FBX, which Blender's importer flatly
rejects ("ASCII FBX files are not supported"). Binary FBX 7.4 / version 7400 is the
most widely compatible binary flavour — Blender, 3ds Max / Maya 2013+, MotionBuilder,
Unity and Unreal all import it. Offsets are uint32 (7.5 switched them to uint64).

The file is a tree of records (nodes). Each record:
    uint32 EndOffset        # absolute byte offset where this record (incl. children) ends
    uint32 NumProperties
    uint32 PropertyListLen   # byte length of the encoded property list
    uint8  NameLen
    char   Name[NameLen]
    <properties>
    <nested records>         # optional; if present, terminated by a 13-byte null record

Property type codes: Y int16, C bool, I int32, F float32, D float64, L int64,
S string (uint32 len + bytes), R raw, and array codes f/d/i/l (uint32 count,
uint32 encoding[0=raw,1=deflate], uint32 byteLen, data).

We self-validate every file we write by re-parsing it (see roundtrip_ok) so a
structural mistake fails loudly instead of producing a file that silently won't import.
"""

import struct
import zlib
import math
import time

# FBX time unit: KTime = seconds * FBX_KTIME
FBX_KTIME = 46186158000
# Final 16-byte magic that closes every binary FBX (constant across files).
_FOOTER_MAGIC = bytes([0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e,
                       0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b])


def _quat_mul(a, b):
    """Hamilton product a*b, quaternions as [w,x,y,z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


# COLMAP cameras look down +Z with +Y down. The FBX standard camera aims down its local
# +X axis with +Y up (every compliant importer -- Blender/Maya/Max/Nuke -- compensates for
# this on load), so we map the camera's local axes to: +X -> COLMAP look (+Z), +Y -> up
# (-Y_colmap), +Z -> right (+X_colmap). That mapping is a 180 deg turn about the (1,0,1)
# axis = quaternion [0, v, 0, v] with v = sqrt(1/2). Right-multiplying the camera-to-world
# rotation by it re-aims the camera without moving it, so positions stay COLMAP-frame
# (matching the splat) and the camera faces correctly in every DCC.
_SQRT1_2 = math.sqrt(0.5)
_CAM_LOOK_FLIP = [0.0, _SQRT1_2, 0.0, _SQRT1_2]


def _to_dcc_camera_quat(q):
    """COLMAP camera-to-world quaternion [w,x,y,z] -> DCC camera-orientation quaternion."""
    return _quat_mul(q, _CAM_LOOK_FLIP)


def _quaternion_to_euler_deg(q):
    """[w,x,y,z] -> (X,Y,Z) Euler degrees, XYZ order. Matches the old ASCII path."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


# --------------------------------------------------------------------------- #
# Low-level node tree
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name):
        self.name = name
        self.props = []      # list of (typecode, value)
        self.children = []

    def prop(self, typecode, value):
        self.props.append((typecode, value))
        return self

    def add(self, name):
        n = Node(name)
        self.children.append(n)
        return n

    # convenience: an FBX "P: name, type, subtype, flags, values..." property record
    def p70(self, name, ptype, subtype, flags, *values):
        n = self.add("P")
        n.prop("S", name).prop("S", ptype).prop("S", subtype).prop("S", flags)
        for tc, v in values:
            n.prop(tc, v)
        return n


def _encode_prop(typecode, value):
    if typecode == "Y":
        return b"Y" + struct.pack("<h", int(value))
    if typecode == "C":
        return b"C" + struct.pack("<?", bool(value))
    if typecode == "I":
        return b"I" + struct.pack("<i", int(value))
    if typecode == "F":
        return b"F" + struct.pack("<f", float(value))
    if typecode == "D":
        return b"D" + struct.pack("<d", float(value))
    if typecode == "L":
        return b"L" + struct.pack("<q", int(value))
    if typecode == "S":
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return b"S" + struct.pack("<I", len(data)) + data
    if typecode == "R":
        return b"R" + struct.pack("<I", len(value)) + bytes(value)
    if typecode in ("f", "d", "i", "l"):
        fmt = {"f": "f", "d": "d", "i": "i", "l": "q"}[typecode]
        raw = struct.pack("<%d%s" % (len(value), fmt), *value)
        comp = zlib.compress(raw)
        if len(comp) < len(raw):
            return typecode.encode() + struct.pack("<III", len(value), 1, len(comp)) + comp
        return typecode.encode() + struct.pack("<III", len(value), 0, len(raw)) + raw
    raise ValueError("unknown property type %r" % typecode)


def _serialize(node, start):
    """Serialize node at absolute offset `start`; returns its bytes (EndOffset correct)."""
    name = node.name.encode("utf-8")
    props = b"".join(_encode_prop(tc, v) for tc, v in node.props)
    header_len = 13 + len(name)          # 3*uint32 + uint8 namelen + name
    cursor = start + header_len + len(props)
    children = b""
    for ch in node.children:
        cb = _serialize(ch, cursor)
        children += cb
        cursor += len(cb)
    if node.children:                    # nested list is null-terminated
        children += b"\x00" * 13
        cursor += 13
    end_offset = cursor
    out = struct.pack("<III", end_offset, len(node.props), len(props))
    out += struct.pack("<B", len(name)) + name + props + children
    return out


def _build_file(top_nodes):
    """Assemble the full binary FBX byte string from a list of top-level nodes."""
    header = b"Kaydara FBX Binary  \x00" + b"\x1a\x00" + struct.pack("<I", 7400)
    body = b""
    cursor = len(header)
    for n in top_nodes:
        b = _serialize(n, cursor)
        body += b
        cursor += len(b)
    body += b"\x00" * 13                 # top-level null record
    cursor += 13
    # Footer: 16-byte file id (zeros are accepted by importers) + zero-pad to 16-byte
    # alignment + uint32 0 + version + 120 zeros + magic.
    footer = b"\x00" * 16
    total = cursor + len(footer)
    pad = (16 - (total % 16)) % 16
    footer += b"\x00" * pad
    footer += struct.pack("<I", 0)
    footer += struct.pack("<I", 7400)
    footer += b"\x00" * 120
    footer += _FOOTER_MAGIC
    return header + body + footer


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #
_ID = [1000000]


def _uid():
    _ID[0] += 1
    return _ID[0]


def _objname(name, cls):
    """Binary-FBX object name property: b"Name\\x00\\x01Class".

    Binary FBX joins the object name and its class with a 0x00,0x01 separator
    (Blender does props[-2].split(b'\\x00\\x01')). This is NOT the ASCII-FBX
    "Class::Name" form with literal '::' — using that crashes Blender's importer.
    """
    return name + "\x00\x01" + cls


def _header_extension():
    h = Node("FBXHeaderExtension")
    h.add("FBXHeaderVersion").prop("I", 1003)
    h.add("FBXVersion").prop("I", 7400)
    t = time.localtime()
    ct = h.add("CreationTimeStamp")
    ct.add("Version").prop("I", 1000)
    ct.add("Year").prop("I", t.tm_year)
    ct.add("Month").prop("I", t.tm_mon)
    ct.add("Day").prop("I", t.tm_mday)
    ct.add("Hour").prop("I", t.tm_hour)
    ct.add("Minute").prop("I", t.tm_min)
    ct.add("Second").prop("I", t.tm_sec)
    ct.add("Millisecond").prop("I", 0)
    h.add("Creator").prop("S", "images_to_play camera tracker")
    return h


def _fps_to_timemode(fps):
    """Map fps -> (FBX TimeMode enum, custom_rate). Standard rates use their named enum
    (most compatible); anything else uses eCustom (14) + CustomFrameRate."""
    named = {120: 1, 100: 2, 60: 3, 50: 4, 48: 5, 30: 6, 25: 10, 24: 11}
    return (named.get(int(round(fps)), 14), float(fps))


def _global_settings(fps=30):
    g = Node("GlobalSettings")
    g.add("Version").prop("I", 1000)
    props = g.add("Properties70")
    # Z-up / -Y-forward / X-right == Blender's native axes, so Blender imports with
    # NO reorientation and the camera stays in the same frame as the (axis-less) splat
    # PLY. (The old Y-up declaration made Blender rotate everything 90 deg about X.)
    props.p70("UpAxis", "int", "Integer", "", ("I", 2))
    props.p70("UpAxisSign", "int", "Integer", "", ("I", 1))
    props.p70("FrontAxis", "int", "Integer", "", ("I", 1))
    props.p70("FrontAxisSign", "int", "Integer", "", ("I", -1))
    props.p70("CoordAxis", "int", "Integer", "", ("I", 0))
    props.p70("CoordAxisSign", "int", "Integer", "", ("I", 1))
    # 1 file unit = 1 metre. FBX is centimetre-native, so UnitScaleFactor=100 makes a
    # metre-based DCC (Blender) import positions 1:1 instead of 100x too small.
    props.p70("UnitScaleFactor", "double", "Number", "", ("D", 100.0))
    props.p70("OriginalUnitScaleFactor", "double", "Number", "", ("D", 100.0))
    # Declare the frame rate so keyframes (authored at `fps`) land on integer frames in
    # the DCC instead of being resampled to the app's default rate.
    timemode, custom_rate = _fps_to_timemode(fps)
    props.p70("TimeMode", "enum", "", "", ("I", timemode))
    props.p70("CustomFrameRate", "double", "Number", "", ("D", custom_rate))
    return g


def _definitions(num_curvenodes, num_curves):
    d = Node("Definitions")
    d.add("Version").prop("I", 100)
    # GlobalSettings(1) + Model(1) + NodeAttribute(1) + AnimStack(1) + AnimLayer(1)
    total = 5 + num_curvenodes + num_curves
    d.add("Count").prop("I", total)

    def obj(name, count):
        o = d.add("ObjectType")
        o.prop("S", name)
        o.add("Count").prop("I", count)
        return o

    obj("GlobalSettings", 1)
    obj("Model", 1)
    obj("NodeAttribute", 1)
    obj("AnimationStack", 1)
    obj("AnimationLayer", 1)
    obj("AnimationCurveNode", num_curvenodes)
    obj("AnimationCurve", num_curves)
    return d


def _camera_model(model_id, name, t0, r0):
    m = Node("Model")
    m.prop("L", model_id).prop("S", _objname(name, "Model")).prop("S", "Camera")
    m.add("Version").prop("I", 232)
    p = m.add("Properties70")
    p.p70("Lcl Translation", "Lcl Translation", "", "A",
          ("D", t0[0]), ("D", t0[1]), ("D", t0[2]))
    p.p70("Lcl Rotation", "Lcl Rotation", "", "A",
          ("D", r0[0]), ("D", r0[1]), ("D", r0[2]))
    p.p70("DefaultAttributeIndex", "int", "Integer", "", ("I", 0))
    m.add("Shading").prop("C", True)
    m.add("Culling").prop("S", "CullingOff")
    return m


def _camera_attr(attr_id, focal_mm):
    a = Node("NodeAttribute")
    a.prop("L", attr_id).prop("S", _objname("", "NodeAttribute")).prop("S", "Camera")
    a.add("TypeFlags").prop("S", "Camera")
    a.add("GeometryVersion").prop("I", 124)
    p = a.add("Properties70")
    p.p70("FocalLength", "double", "Number", "A", ("D", float(focal_mm)))
    p.p70("FilmWidth", "double", "Number", "", ("D", 36.0 / 25.4))
    p.p70("FilmHeight", "double", "Number", "", ("D", 24.0 / 25.4))
    return a


def _curve(curve_id, times, values):
    c = Node("AnimationCurve")
    c.prop("L", curve_id).prop("S", _objname("", "AnimCurve")).prop("S", "")
    c.add("Default").prop("D", float(values[0]) if values else 0.0)
    c.add("KeyVer").prop("I", 4009)
    c.add("KeyTime").prop("l", times)
    c.add("KeyValueFloat").prop("f", [float(v) for v in values])
    n = len(times)
    # constant-ish flags (2 = interpolation cubic|auto); importers re-fit anyway
    c.add("KeyAttrFlags").prop("i", [24840])
    c.add("KeyAttrDataFloat").prop("f", [0.0, 0.0, 0.0, 0.0])
    c.add("KeyAttrRefCount").prop("i", [n])
    return c


def _curve_node(node_id, label, dx, dy, dz):
    cn = Node("AnimationCurveNode")
    cn.prop("L", node_id).prop("S", _objname(label, "AnimCurveNode")).prop("S", "")
    p = cn.add("Properties70")
    p.p70("d|X", "Number", "", "A", ("D", dx))
    p.p70("d|Y", "Number", "", "A", ("D", dy))
    p.p70("d|Z", "Number", "", "A", ("D", dz))
    return cn


def write_camera_fbx(poses, output_path, fps=30, frame_numbers=None):
    """Write `poses` (from extract_camera_poses) as a binary FBX 7.4 animated camera.

    `frame_numbers` optionally gives each pose's source-frame number (e.g. for video
    decimated by N, [0, N, 2N, ...]) so the camera lines up with the original footage at
    `fps`. Defaults to consecutive frames 0..len(poses)-1.
    """
    if not poses:
        raise ValueError("no poses to export")

    model_id = _uid()
    attr_id = _uid()
    stack_id = _uid()
    layer_id = _uid()

    # focal length: pixel -> mm (36mm sensor), from frame 0
    p0 = poses[0]
    fpx = p0.get("focal_length", 0) or 0
    iw = p0.get("width", 0) or 1920
    focal_mm = (36.0 * fpx / iw) if (fpx > 0 and iw > 0) else 35.0

    t0 = poses[0]["position"]
    r0 = _quaternion_to_euler_deg(_to_dcc_camera_quat(poses[0]["rotation_quat"]))

    # animation samples -- key i sits at its source-frame number so a video-derived
    # camera lines up with the original footage; defaults to consecutive frames.
    if frame_numbers is None:
        frame_numbers = list(range(len(poses)))
    times = [int(round(fn / fps * FBX_KTIME)) for fn in frame_numbers]
    tx = [p["position"][0] for p in poses]
    ty = [p["position"][1] for p in poses]
    tz = [p["position"][2] for p in poses]
    eul = [_quaternion_to_euler_deg(_to_dcc_camera_quat(p["rotation_quat"])) for p in poses]
    rx = [e[0] for e in eul]
    ry = [e[1] for e in eul]
    rz = [e[2] for e in eul]

    # objects
    objects = Node("Objects")
    objects.children.append(_camera_model(model_id, "TrackingCamera", t0, r0))
    objects.children.append(_camera_attr(attr_id, focal_mm))

    stack = objects.add("AnimationStack")
    stack.prop("L", stack_id).prop("S", _objname("Take 001", "AnimStack")).prop("S", "")
    layer = objects.add("AnimationLayer")
    layer.prop("L", layer_id).prop("S", _objname("BaseLayer", "AnimLayer")).prop("S", "")

    t_node_id, r_node_id = _uid(), _uid()
    objects.children.append(_curve_node(t_node_id, "T", t0[0], t0[1], t0[2]))
    objects.children.append(_curve_node(r_node_id, "R", r0[0], r0[1], r0[2]))

    curves = []  # (curve_id, parent_curvenode_id, channel, data)
    for chan, data, parent in (
        ("d|X", tx, t_node_id), ("d|Y", ty, t_node_id), ("d|Z", tz, t_node_id),
        ("d|X", rx, r_node_id), ("d|Y", ry, r_node_id), ("d|Z", rz, r_node_id),
    ):
        cid = _uid()
        objects.children.append(_curve(cid, times, data))
        curves.append((cid, parent, chan))

    # connections
    conns = Node("Connections")

    def oo(child, parent):
        c = conns.add("C")
        c.prop("S", "OO").prop("L", child).prop("L", parent)

    def op(child, parent, prop_name):
        c = conns.add("C")
        c.prop("S", "OP").prop("L", child).prop("L", parent).prop("S", prop_name)

    oo(model_id, 0)               # camera under scene root
    oo(attr_id, model_id)         # attribute on camera
    oo(layer_id, stack_id)        # layer in stack
    op(t_node_id, model_id, "Lcl Translation")
    op(r_node_id, model_id, "Lcl Rotation")
    oo(t_node_id, layer_id)
    oo(r_node_id, layer_id)
    for cid, parent, chan in curves:
        op(cid, parent, chan)

    top = [
        _header_extension(),
        _global_settings(fps),
        _definitions(num_curvenodes=2, num_curves=len(curves)),
        objects,
        conns,
    ]
    data = _build_file(top)

    # self-validate before writing: a structural error must fail here, not in Blender
    if not roundtrip_ok(data):
        raise RuntimeError("internal FBX serialization failed self-validation")

    with open(output_path, "wb") as f:
        f.write(data)
    return output_path


# --------------------------------------------------------------------------- #
# Self-validation: re-parse the record tree and confirm every EndOffset lines up
# --------------------------------------------------------------------------- #
def roundtrip_ok(data):
    try:
        if data[:21] != b"Kaydara FBX Binary  \x00":
            return False
        ver = struct.unpack("<I", data[23:27])[0]
        if ver != 7400:
            return False
        pos = 27

        def parse_record(p):
            end, num_props, prop_len = struct.unpack("<III", data[p:p + 12])
            if end == 0:
                return 0, p + 13      # null record
            name_len = data[p + 12]
            q = p + 13 + name_len + prop_len
            # walk nested records (if any) until we reach `end`
            while q < end:
                child_end, q = parse_record(q)
                if child_end == 0:
                    break
            if q != end:
                raise ValueError("offset mismatch %d != %d" % (q, end))
            return end, end

        while pos < len(data):
            end, nxt = parse_record(pos)
            if end == 0:
                break
            pos = nxt
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # quick smoke test with a synthetic 10-frame orbit
    import os
    poses = []
    for i in range(10):
        a = i / 10 * math.pi
        poses.append({
            "position": [math.cos(a), math.sin(a), 0.5 * i],
            "rotation_quat": [math.cos(a / 2), 0, 0, math.sin(a / 2)],
            "focal_length": 1600, "width": 1920, "height": 1080,
        })
    out = write_camera_fbx(poses, "test_camera.fbx", fps=30)
    print("wrote", out, os.path.getsize(out), "bytes; roundtrip OK")
