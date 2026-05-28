"""
Batch Processing Module for images_to_play

Handles multiple job submissions, queue management, and batch status tracking.
"""
import os
import uuid
import time
import json
import threading
from collections import OrderedDict
from pathlib import Path


class BatchJob:
    """Represents a single job within a batch."""
    def __init__(self, job_id, name, preset, settings=None):
        self.job_id = job_id
        self.name = name
        self.preset = preset
        self.settings = settings or {}
        self.status = 'pending'  # pending, uploading, running, complete, failed, cancelled
        self.progress = 0.0
        self.stage = ''
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.output_ply = None

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'name': self.name,
            'preset': self.preset,
            'status': self.status,
            'progress': self.progress,
            'stage': self.stage,
            'error': self.error,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'elapsed': (self.completed_at or time.time()) - (self.started_at or self.created_at),
            'output_ply': self.output_ply,
        }


class BatchQueue:
    """Manages a queue of jobs for batch processing."""

    def __init__(self, max_concurrent=3):
        self.max_concurrent = max_concurrent
        self._batches = OrderedDict()  # batch_id -> {info, jobs: [BatchJob]}
        self._lock = threading.Lock()
        self._processing_thread = None
        self._stop_event = threading.Event()

    def create_batch(self, name=None):
        """Create a new batch and return its ID."""
        batch_id = str(uuid.uuid4())
        with self._lock:
            self._batches[batch_id] = {
                'id': batch_id,
                'name': name or f'Batch {batch_id[:8]}',
                'created_at': time.time(),
                'status': 'pending',  # pending, running, complete, partial
                'jobs': [],
            }
        return batch_id

    def add_job(self, batch_id, name, preset, image_paths, settings=None):
        """Add a job to a batch. Returns the job_id."""
        if batch_id not in self._batches:
            raise ValueError(f"Batch {batch_id} not found")

        job_id = str(uuid.uuid4())
        job = BatchJob(job_id, name, preset, settings)
        job._image_paths = image_paths  # stored for processing

        with self._lock:
            self._batches[batch_id]['jobs'].append(job)

        return job_id

    def start_batch(self, batch_id, upload_fn, process_fn):
        """Start processing a batch. Runs in background thread."""
        if batch_id not in self._batches:
            raise ValueError(f"Batch {batch_id} not found")

        with self._lock:
            self._batches[batch_id]['status'] = 'running'

        thread = threading.Thread(
            target=self._process_batch,
            args=(batch_id, upload_fn, process_fn),
            daemon=True
        )
        thread.start()
        return thread

    def _process_batch(self, batch_id, upload_fn, process_fn):
        """Process all jobs in a batch sequentially."""
        batch = self._batches[batch_id]
        jobs = batch['jobs']

        for job in jobs:
            if self._stop_event.is_set():
                job.status = 'cancelled'
                continue

            try:
                job.status = 'running'
                job.started_at = time.time()

                # Upload images and get job_id from the main app
                app_job_id = upload_fn(job._image_paths, job.preset, job.settings)
                job.app_job_id = app_job_id

                # Monitor until completion
                result = process_fn(app_job_id, job)
                job.status = 'complete' if result.get('success') else 'failed'
                job.output_ply = result.get('output_ply')
                job.error = result.get('error')

            except Exception as e:
                job.status = 'failed'
                job.error = str(e)

            finally:
                job.completed_at = time.time()

        # Update batch status
        with self._lock:
            statuses = [j.status for j in jobs]
            if all(s == 'complete' for s in statuses):
                batch['status'] = 'complete'
            elif any(s == 'complete' for s in statuses):
                batch['status'] = 'partial'
            elif all(s in ('failed', 'cancelled') for s in statuses):
                batch['status'] = 'failed'
            else:
                batch['status'] = 'complete'

    def get_batch_status(self, batch_id):
        """Get status of all jobs in a batch."""
        if batch_id not in self._batches:
            return None

        batch = self._batches[batch_id]
        jobs = batch['jobs']

        total = len(jobs)
        completed = sum(1 for j in jobs if j.status == 'complete')
        failed = sum(1 for j in jobs if j.status == 'failed')
        running = sum(1 for j in jobs if j.status == 'running')
        pending = sum(1 for j in jobs if j.status == 'pending')

        # Overall progress
        if total > 0:
            overall_progress = sum(j.progress for j in jobs) / total
        else:
            overall_progress = 0

        # ETA calculation
        elapsed_jobs = [j for j in jobs if j.completed_at and j.started_at]
        if elapsed_jobs and pending > 0:
            avg_time = sum(j.completed_at - j.started_at for j in elapsed_jobs) / len(elapsed_jobs)
            eta = avg_time * pending
        else:
            eta = None

        return {
            'batch_id': batch_id,
            'name': batch['name'],
            'status': batch['status'],
            'total': total,
            'completed': completed,
            'failed': failed,
            'running': running,
            'pending': pending,
            'overall_progress': overall_progress,
            'eta_seconds': eta,
            'jobs': [j.to_dict() for j in jobs],
        }

    def cancel_batch(self, batch_id):
        """Cancel all pending jobs in a batch."""
        if batch_id not in self._batches:
            return False

        with self._lock:
            for job in self._batches[batch_id]['jobs']:
                if job.status in ('pending', 'running'):
                    job.status = 'cancelled'
            self._stop_event.set()

        return True

    def list_batches(self):
        """List all batches with summary info."""
        result = []
        for batch_id, batch in self._batches.items():
            jobs = batch['jobs']
            result.append({
                'batch_id': batch_id,
                'name': batch['name'],
                'status': batch['status'],
                'total_jobs': len(jobs),
                'completed': sum(1 for j in jobs if j.status == 'complete'),
                'created_at': batch['created_at'],
            })
        return result

    def remove_batch(self, batch_id):
        """Remove a completed/failed batch from memory."""
        with self._lock:
            if batch_id in self._batches:
                batch = self._batches[batch_id]
                if batch['status'] in ('complete', 'failed', 'partial'):
                    del self._batches[batch_id]
                    return True
        return False


# Global batch queue instance
batch_queue = BatchQueue(max_concurrent=3)


def register_batch_routes(app, processing_folder, upload_fn, process_fn, monitor_fn):
    """Register batch processing routes on the Flask app."""

    @app.route('/batch/create', methods=['POST'])
    def batch_create():
        """Create a new batch."""
        from flask import request, jsonify
        name = request.json.get('name') if request.is_json else None
        batch_id = batch_queue.create_batch(name)
        return jsonify({'batch_id': batch_id, 'name': name or f'Batch {batch_id[:8]}'})

    @app.route('/batch/<batch_id>/add', methods=['POST'])
    def batch_add_job(batch_id):
        """Add a job to a batch. Expects multipart form with files + settings."""
        from flask import request, jsonify
        from werkzeug.utils import secure_filename

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        preset = request.form.get('preset', 'medium')
        name = request.form.get('name', f'Job {preset}')

        # Save files to temp folder
        job_folder = os.path.join(processing_folder, '_batch_staging', batch_id)
        images_folder = os.path.join(job_folder, secure_filename(name), 'source')
        os.makedirs(images_folder, exist_ok=True)

        saved_paths = []
        for f in files:
            if f.filename:
                fpath = os.path.join(images_folder, secure_filename(f.filename))
                f.save(fpath)
                saved_paths.append(fpath)

        if not saved_paths:
            return jsonify({'error': 'No valid files saved'}), 400

        settings = {
            'matcher_type': request.form.get('matcher_type', 'exhaustive_matcher'),
            'interval': int(request.form.get('interval', 1)),
            'quality_scale': request.form.get('quality_scale', 'standard'),
            'trainer': request.form.get('trainer', 'brush'),
            'enable_dense': request.form.get('enable_dense', 'true').lower() == 'true',
            'training_steps': request.form.get('training_steps'),
            'max_image_size': int(request.form.get('max_image_size', 3200)),
        }

        job_id = batch_queue.add_job(batch_id, name, preset, saved_paths, settings)
        return jsonify({'job_id': job_id, 'name': name, 'preset': preset, 'files': len(saved_paths)})

    @app.route('/batch/<batch_id>/start', methods=['POST'])
    def batch_start(batch_id):
        """Start processing a batch. Uses direct function calls instead of self-HTTP."""
        from flask import jsonify

        def _upload(image_paths, preset, settings):
            """Create a job by calling process_fn directly (no HTTP roundtrip)."""
            import uuid as _uuid
            job_id = str(_uuid.uuid4())
            # Use the first image path's parent as the image folder
            image_folder = os.path.dirname(image_paths[0])
            # Call the process function directly
            if process_fn:
                thread = threading.Thread(
                    target=process_fn,
                    args=(job_id, image_folder, preset, settings),
                    daemon=True
                )
                thread.start()
            return job_id

        def _monitor(app_job_id, batch_job):
            """Monitor a job using direct memory access (no HTTP)."""
            while True:
                try:
                    if monitor_fn:
                        status = monitor_fn(app_job_id)
                    else:
                        status = None

                    if status is None:
                        time.sleep(2)
                        continue

                    batch_job.progress = status.get('progress', 0)
                    batch_job.stage = status.get('stage', '')

                    if status.get('status') == 'completed':
                        return {'success': True, 'output_ply': status.get('ply_path')}
                    elif status.get('status') in ('error', 'failed'):
                        return {'success': False, 'error': status.get('error', 'Unknown error')}
                    elif status.get('status') == 'cancelled':
                        return {'success': False, 'error': 'Cancelled'}

                    time.sleep(3)
                except Exception as e:
                    return {'success': False, 'error': str(e)}

        batch_queue.start_batch(batch_id, _upload, _monitor)
        return jsonify({'status': 'started', 'batch_id': batch_id})

    @app.route('/batch/<batch_id>/status')
    def batch_status(batch_id):
        """Get batch status."""
        from flask import jsonify
        status = batch_queue.get_batch_status(batch_id)
        if status is None:
            return jsonify({'error': 'Batch not found'}), 404
        return jsonify(status)

    @app.route('/batch/<batch_id>/cancel', methods=['POST'])
    def batch_cancel(batch_id):
        """Cancel a batch."""
        from flask import jsonify
        success = batch_queue.cancel_batch(batch_id)
        return jsonify({'cancelled': success})

    @app.route('/batch/list')
    def batch_list():
        """List all batches."""
        from flask import jsonify
        return jsonify(batch_queue.list_batches())

    @app.route('/batch/<batch_id>/remove', methods=['POST'])
    def batch_remove(batch_id):
        """Remove a completed batch."""
        from flask import jsonify
        success = batch_queue.remove_batch(batch_id)
        return jsonify({'removed': success})

    @app.route('/batch/upload-folder', methods=['POST'])
    def batch_upload_folder():
        """
        Upload a folder structure for batch processing.
        Expects a ZIP with subfolders, each subfolder = one job.
        """
        from flask import request, jsonify
        import zipfile
        from werkzeug.utils import secure_filename

        if 'file' not in request.files:
            return jsonify({'error': 'No ZIP file provided'}), 400

        zip_file = request.files['file']
        batch_name = request.form.get('name', 'Folder Batch')
        default_preset = request.form.get('preset', 'medium')

        # Create batch
        batch_id = batch_queue.create_batch(batch_name)

        # Save and extract ZIP
        staging = os.path.join(processing_folder, '_batch_staging', batch_id)
        os.makedirs(staging, exist_ok=True)
        zip_path = os.path.join(staging, 'upload.zip')
        zip_file.save(zip_path)

        # Extract and find subfolders
        extract_dir = os.path.join(staging, 'extracted')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Each subfolder = one job
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        jobs_added = 0

        for entry in sorted(os.listdir(extract_dir)):
            entry_path = os.path.join(extract_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            # Find images in subfolder
            images = []
            for f in sorted(os.listdir(entry_path)):
                if f.lower().endswith(image_exts):
                    images.append(os.path.join(entry_path, f))

            if images:
                batch_queue.add_job(batch_id, entry, default_preset, images)
                jobs_added += 1

        # Also check if images are directly in the ZIP root (single job)
        if jobs_added == 0:
            root_images = []
            for f in sorted(os.listdir(extract_dir)):
                if f.lower().endswith(image_exts):
                    root_images.append(os.path.join(extract_dir, f))

            if root_images:
                batch_queue.add_job(batch_id, 'Root Images', default_preset, root_images)
                jobs_added += 1

        if jobs_added == 0:
            return jsonify({'error': 'No valid image folders found in ZIP'}), 400

        return jsonify({
            'batch_id': batch_id,
            'name': batch_name,
            'jobs': jobs_added,
            'message': f'Batch created with {jobs_added} jobs. POST /batch/{batch_id}/start to begin.'
        })
