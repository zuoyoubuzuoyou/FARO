#!/usr/bin/env python3
"""Losslessly repack a finished V3 episode into a fresh directory, offline.

Usage: compact_trajectory.py --input EPISODE --output FRESH_EPISODE [--report PATH]
The default report is FRESH_EPISODE/compaction_report.json. Report bytes are
excluded from size totals, which include all payloads and media. Legacy episode
metadata is plain JSON except initial_event; preserve that distinction. Pending
or unterminated rows are refused, never discarded. Row-only evidence that cannot
be reconstructed exactly from events is also refused. Source files are SHA256
checked before and after conversion. No simulator or model dependencies load.
"""
import argparse
import hashlib
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys

_STORAGE = Path(__file__).resolve().parents[1] / 'habitat-mas/habitat_mas/utils/trajectory_storage.py'
_spec = importlib.util.spec_from_file_location('_offline_trajectory_storage', _STORAGE)
storage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(storage)


def _inside(path, directory):
    return path == directory or directory in path.parents


def _run_root(episode):
    return episode.parent.parent if episode.parent.name == 'episodes' else episode


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks are not supported in source data: %s' % path)
        if path.is_file():
            files[str(path.relative_to(directory))] = _digest(path)
    return files


def _stats(directory, hashes):
    return {'directory_bytes': sum((directory / name).stat().st_size for name in hashes),
            'steps_bytes': (directory / 'steps.jsonl').stat().st_size,
            'file_count': len(hashes), 'sha256': hashes}


def _json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        try:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
        except BaseException:
            path.unlink()
            raise


def _rows(path):
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                raise ValueError('Uncommitted/truncated trailing row at line %d' % number)
            if line.strip():
                row = json.loads(line.decode('utf-8-sig' if number == 1 else 'utf-8'))
                if not isinstance(row, dict) or row.get('schema_version') != 3 or not isinstance(row.get('events'), list):
                    raise ValueError('Input is not a V3 step packet at line %d' % number)
                yield row


def _run_files(source, output, metadata):
    """Plan only explicit config links and optional sibling run metadata."""
    files = metadata.get('files', {})
    link = files.get('config') if isinstance(files, dict) else None
    copies = []
    if link is not None:
        if not isinstance(link, str) or not link or Path(link).is_absolute():
            raise ValueError('Config link must be relative')
        linked_source = source / link
        if linked_source.is_symlink():
            raise ValueError('Symlinked config is not supported')
        original = linked_source.resolve()
        destination = (output / link).resolve()
        if not _inside(original, _run_root(source)) or not _inside(destination, _run_root(output)):
            raise ValueError('Config link escapes the corresponding run directory')
        if original.is_symlink() or not original.is_file():
            raise ValueError('Missing or unsafe linked config: %s' % original)
        copies.append((original, destination))
    if _run_root(source) != source:
        original = _run_root(source) / 'run.json'
        if original.exists():
            if original.is_symlink() or not original.is_file():
                raise ValueError('Unsafe run metadata')
            copies.append((original, _run_root(output) / 'run.json'))
    return copies


def compact(source, output, report_path=None):
    """Return a verified report; remove only newly created output on failure."""
    source, output = Path(source), Path(output)
    report_path = Path(report_path) if report_path else output / 'compaction_report.json'
    if source.is_symlink() or output.is_symlink() or report_path.is_symlink():
        raise ValueError('Input, output and report paths must not be symlinks')
    source, output, report_path = source.resolve(), output.resolve(), report_path.resolve()
    if not source.is_dir():
        raise ValueError('Input must be an episode directory')
    source_root = _run_root(source)
    if output.exists() or _inside(output, source_root) or _inside(source, output):
        raise ValueError('Output must be fresh and outside the source run directory')
    if report_path.exists() or _inside(report_path, source_root):
        raise ValueError('Report must be fresh and outside the source run directory')
    if (source / 'pending.json').exists():
        raise ValueError('Pending evidence exists; only finished episodes can be compacted')
    metadata = _json(source / 'episode.json')
    if not isinstance(metadata, dict) or metadata.get('storage_schema_version', metadata.get('schema_version')) != 3:
        raise ValueError('Only V3 episode directories are supported')
    if not (source / 'summary.json').is_file():
        raise ValueError('Finished episode summary is required')
    # Validate the committed file tail before creating anything.
    with (source / 'steps.jsonl').open('rb') as stream:
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b'\n':
                raise ValueError('Uncommitted/truncated trailing row')
    before = _manifest(source)
    linked = _run_files(source, output, metadata)
    linked_hashes = {str(original): _digest(original) for original, _ in linked}
    for original, destination in linked:
        if _inside(destination, source_root):
            raise ValueError('Copied run files would modify the source')
        if destination.exists() and (not destination.is_file() or _digest(destination) != linked_hashes[str(original)]):
            raise ValueError('Refusing to overwrite existing run file: %s' % destination)
        if destination == report_path:
            raise ValueError('Report collides with a linked run file')
    if _inside(report_path, output) and str(report_path.relative_to(output)) in before:
        raise ValueError('Report collides with source evidence')
    created = False
    created_links = []
    created_report = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        codec = storage._Codec(source)
        # The recorder writes context/initial_state as plain JSON. Decoding the
        # entire metadata would reinterpret literal user $trace dictionaries.
        static_metadata = {key: value for key, value in metadata.items() if key != 'initial_event'}
        _write_json(output / 'episode.json', static_metadata)
        store = storage.StepTraceStore(output, metadata)
        if 'initial_event' in metadata:
            store.append(codec.decode(metadata['initial_event']))
        rows = 0
        for row in _rows(source / 'steps.jsonl'):
            rows += 1
            if row.get('step') != rows:
                raise ValueError('Non-contiguous source row')
            for event in row['events']:
                store.append(codec.decode(event, row))
            store.commit_step()
            if store.step != rows:
                raise ValueError('Source row does not contain one returned step')
            # Public V3 readers accept canonical/extra evidence that appears
            # only on the row. Event equality alone cannot detect its loss.
            # Refuse unsupported rows rather than discard that evidence.
            for key, value in row.items():
                if key in ('events', 'schema_version', 'step'):
                    continue
                if key not in store.last_row:
                    raise ValueError('Cannot preserve row-only field %s at step %d' % (key, rows))
                original_value = codec.decode(value, row)
                compacted_value = store.codec.decode(store.last_row[key], store.last_row)
                if json.dumps(original_value, sort_keys=True, ensure_ascii=False) != json.dumps(compacted_value, sort_keys=True, ensure_ascii=False):
                    raise ValueError('Expanded row field %s differs at step %d' % (key, rows))
        summary = storage.read_trace_summary(source)
        store.write_summary(summary)
        store.close()
        for name in before:
            # Re-encoding creates only payloads actually used by the new store.
            if name.split('/')[0] in ('payloads', 'episode.json', 'steps.jsonl', 'summary.json'):
                continue
            original = source / name
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with original.open('rb') as reader, destination.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
            if _digest(destination) != before[name]:
                raise ValueError('Copied evidence differs: %s' % name)
        for original, destination in linked:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with original.open('rb') as reader, destination.open('xb') as writer:
                    created_links.append(destination)
                    shutil.copyfileobj(reader, writer)
            if _digest(destination) != linked_hashes[str(original)]:
                raise ValueError('Copied config/run metadata differs')
        # These public iterators independently validate packet identity,
        # canonical fields, sequence and expansion while comparison streams.
        missing = object()
        events = 0
        for original, compacted in zip_longest(storage.iter_trace_events(source), storage.iter_trace_events(output), fillvalue=missing):
            if original != compacted:
                raise ValueError('Expanded events differ at event %d' % events)
            events += 1
        if storage.read_trace_summary(output) != summary:
            raise ValueError('Expanded summary differs')
        output_metadata = _json(output / 'episode.json')
        for key, value in static_metadata.items():
            if key == 'storage_features' and isinstance(value, list):
                advertised = output_metadata.get(key, [])
                if isinstance(advertised, list) and all(feature in advertised for feature in value):
                    continue
            if output_metadata.get(key, missing) != value:
                raise ValueError('Episode metadata differs: %s' % key)
        after = _manifest(source)
        if after != before or any(_digest(Path(path)) != digest for path, digest in linked_hashes.items()):
            raise ValueError('Source hashes changed during compaction')
        output_hashes = _manifest(output)
        report = {
            'input_path': str(source), 'output_path': str(output), 'report_path': str(report_path),
            'rows': rows, 'events': events,
            'input': _stats(source, before), 'output': _stats(output, output_hashes),
            'linked_run_files': [{'input': str(original), 'output': str(destination),
                                  'bytes': original.stat().st_size, 'sha256': linked_hashes[str(original)]}
                                 for original, destination in linked],
            'verification': {'events_equal': True, 'summary_equal': True,
                             'metadata_equal': True, 'source_hashes_unchanged': True,
                             'physical_row_boundaries_preserved': True, 'row_fields_equal': True},
            'size_accounting': 'Episode totals include payloads/media; exclude the report. Run totals include the whole corresponding run; exclude the report.'}
        for directory, stats in ((source, report['input']), (output, report['output'])):
            root = _run_root(directory)
            run_paths = [p for p in root.rglob('*') if p.is_file()]
            stats['run_directory_bytes'] = sum(p.stat().st_size for p in run_paths)
            stats['run_file_count'] = len(run_paths)
        report['steps_reduction_bytes'] = report['input']['steps_bytes'] - report['output']['steps_bytes']
        report['directory_reduction_bytes'] = report['input']['directory_bytes'] - report['output']['directory_bytes']
        _write_json(report_path, report)
        created_report = True
        return report
    except BaseException:
        if created_report and report_path.exists():
            report_path.unlink()
        if created:
            shutil.rmtree(output)
        for path in reversed(created_links):
            if path.exists() and not _inside(path, output):
                path.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='Finished V3 episode directory (read only)')
    parser.add_argument('--output', required=True, type=Path, help='Fresh output episode directory')
    parser.add_argument('--report', type=Path, help='Fresh report path; defaults to OUTPUT/compaction_report.json')
    args = parser.parse_args(argv)
    try:
        report = compact(args.input, args.output, args.report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('Compaction refused or failed: %s' % exc, file=sys.stderr)
        return 2
    print(json.dumps({'output': report['output_path'], 'report': report['report_path'],
                      'rows': report['rows'], 'events': report['events'],
                      'input_bytes': report['input']['directory_bytes'],
                      'output_bytes': report['output']['directory_bytes']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
