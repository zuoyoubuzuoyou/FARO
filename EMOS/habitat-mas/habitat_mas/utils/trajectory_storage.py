"""Dependency-free, lossless environment-step storage and legacy readers.

Callers must redact secrets before append/write_summary. Reference envelopes are
escaped for literal JSON, so arbitrary model messages cannot become references.
Only a newline-terminated V3 row is a committed environment transition.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

IDENTITY_KEYS = ('run_id', 'scene_id', 'episode_id', 'evaluation_id', 'env_index')
CONTROL = '$trace'
_STATE_SHARE_BYTES = 192
_EVENT_FIELDS = ('event_type', 'agent_id', 'sequence', 'env_step', 'event_id', 'timestamp')


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _atomic(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Unreadable trajectory file: %s' % path) from exc


def _identity(value, identity):
    if not isinstance(value, dict) or not isinstance(identity, dict):
        raise ValueError('Trajectory identity/envelope must be an object')
    for key in IDENTITY_KEYS:
        if key in value and key in identity and value[key] != identity[key]:
            raise ValueError('Mixed trajectory identity: %s' % key)


def _index(value, path, result):
    if isinstance(value, (dict, list)) and value:
        result.setdefault(_hash(value), path)
    if isinstance(value, dict):
        for key, child in value.items():
            _index(child, path + [key], result)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _index(child, path + [i], result)


def _packet(packet, label):
    if (not isinstance(packet, dict) or packet.get('schema_version') != 3
            or type(packet.get('step')) is not int or packet['step'] < 1
            or not isinstance(packet.get('identity'), dict)
            or not isinstance(packet.get('events'), list)):
        raise ValueError('Invalid trajectory %s packet' % label)
    return packet


def _returned_steps(events, number, required=False):
    if any(not isinstance(event, dict) for event in events):
        raise ValueError('Trajectory packet events must be objects')
    steps = [e for e in events if e.get('event_type', e.get('event')) == 'step']
    if len(steps) > 1 or (required and len(steps) != 1):
        raise ValueError('Invalid number of returned environment steps in packet')
    for event in steps:
        for key in ('env_step', 'step'):
            if key in event and (type(event[key]) is not int or event[key] != number):
                raise ValueError('Non-contiguous returned environment step or conflicting step aliases')
    return steps


def _event_defaults(row):
    """Only explicit, complete committed-row context can supply event fields."""
    _packet(row, 'event context')
    if any(key not in row['identity'] for key in IDENTITY_KEYS):
        raise ValueError('Incomplete compact event identity')
    return dict({key: row['identity'][key] for key in IDENTITY_KEYS},
                schema_version=3, step=row['step'])


def _standard_event(value, defaults):
    return (all(key in value for key in _EVENT_FIELDS)
            and isinstance(value['event_type'], str) and bool(value['event_type'])
            and (value['agent_id'] is None or isinstance(value['agent_id'], str))
            and isinstance(value['event_id'], str) and isinstance(value['timestamp'], str)
            and type(value['sequence']) is int
            and type(value['env_step']) is int and value['env_step'] == defaults['step'])


class _Codec:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._step_cache = None

    def _referenced_step(self, number):
        path = self.directory / 'steps.jsonl'
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._step_cache is not None and self._step_cache[:2] == (signature, number):
            return self._step_cache[2]
        row = next((r for r in (_packet(r, 'step') for r in _rows(path, True))
                    if r['step'] == number), None)
        if row is not None:
            _identity(row['identity'], _read(self.directory / 'episode.json'))
        # One encoded row, never an episode history or shared decoded objects.
        self._step_cache = (signature, number, row)
        return row

    def _blob(self, encoded):
        digest = _hash(encoded)
        reference = {CONTROL: ['blob', digest]}
        if len(_bytes(reference)) >= len(_bytes(encoded)):
            return encoded
        path = self.directory / 'payloads' / (digest + '.json')
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            _atomic(path, encoded)
        return reference

    def encode_state(self, value, local, previous):
        """Share stable subtrees, without materializing unique frame wrappers."""
        candidates = {}

        def collect(child, path):
            if isinstance(child, (dict, list, str)):
                data = _bytes(child)
                if len(data) >= _STATE_SHARE_BYTES:
                    candidates[path] = hashlib.sha256(data).hexdigest()
            if isinstance(child, dict):
                for key, item in child.items():
                    collect(item, path + (key,))
            elif isinstance(child, list):
                for index, item in enumerate(child):
                    collect(item, path + (index,))

        def encode(child, path):
            # State, the agents map and each agent remain directly inspectable.
            inline = not path or path == ('agents',) or (len(path) == 2 and path[0] == 'agents')
            digest = candidates.get(path)
            if not inline and digest and digest in local:
                reference = {CONTROL: ['local', local[digest], digest]}
                if len(_bytes(reference)) < len(_bytes(child)):
                    return reference
            if not inline and digest and (previous.get(path) == digest or path[-1] == 'camera_sensors'):
                # Encode an entire shared block once, avoiding one file per leaf.
                return self._blob(self.encode(child, blobs=False))
            if isinstance(child, dict):
                pairs = {key: encode(item, path + (key,)) for key, item in child.items()}
                return {CONTROL: ['literal', pairs]} if CONTROL in child else pairs
            if isinstance(child, list):
                return [encode(item, path + (index,)) for index, item in enumerate(child)]
            return child

        collect(value, ())
        return encode(value, ()), candidates

    def encode_event(self, event, local, row):
        try:
            defaults = _event_defaults(row)
        except ValueError:
            return self.encode(event, local, root_inline=True)
        if (CONTROL not in event and _standard_event(event, defaults)
                and all(key in event and _bytes(event[key]) == _bytes(value) for key, value in defaults.items())
                and event.get('event') == event['event_type']):
            sparse = {key: value for key, value in event.items() if key not in defaults and key != 'event'}
            # Never collapse the event root into a local ref. The raw fields
            # are also consumed before decode by existing report generators.
            encoded = {key: value if key in _EVENT_FIELDS else self.encode(value, local)
                       for key, value in sparse.items()}
            encoded[CONTROL] = ['event', _hash(defaults)]
            return encoded
        return self.encode(event, local, root_inline=True)

    def encode(self, value, local=None, blobs=True, root_inline=False):
        digest = _hash(value)
        if local and digest in local:
            reference = {CONTROL: ['local', local[digest], digest]}
            if len(_bytes(reference)) < len(_bytes(value)):
                return reference
        if isinstance(value, dict):
            pairs = {k: self.encode(v, local, blobs) for k, v in value.items()}
            encoded = {CONTROL: ['literal', pairs]} if CONTROL in value else pairs
        elif isinstance(value, list):
            encoded = [self.encode(v, local, blobs) for v in value]
        else:
            encoded = value
        if blobs and not root_inline and isinstance(value, (dict, list, str)) and len(_bytes(encoded)) >= 256:
            return self._blob(encoded)
        return encoded

    def decode(self, value, row=None, active=None):
        active = set() if active is None else active
        if isinstance(value, list):
            return [self.decode(v, row, active) for v in value]
        if not isinstance(value, dict):
            return value
        if CONTROL not in value:
            return {k: self.decode(v, row, active) for k, v in value.items()}
        control = value[CONTROL]
        if isinstance(control, list) and len(control) == 2 and control[0] == 'event':
            defaults = _event_defaults(row)
            if (control[1] != _hash(defaults) or any(key in value for key in defaults)
                    or 'event' in value or not _standard_event(value, defaults)):
                raise ValueError('Invalid compact event envelope/context')
            fields = {key: self.decode(child, row, active) for key, child in value.items() if key != CONTROL}
            return dict(defaults, event=fields['event_type'], **fields)
        if len(value) != 1 or not isinstance(control, list) or not control:
            raise ValueError('Invalid trajectory reference envelope')
        kind = control[0]
        if kind == 'literal' and len(control) == 2 and isinstance(control[1], dict):
            return {k: self.decode(v, row, active) for k, v in control[1].items()}
        token = _hash(control)
        if token in active:
            raise ValueError('Cyclic trajectory reference')
        active = active | {token}
        if kind == 'blob' and len(control) == 2:
            digest = control[1]
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Invalid payload hash/path')
            path = self.directory / 'payloads' / (digest + '.json')
            if path.resolve().parent != self.directory.resolve() / 'payloads':
                raise ValueError('Payload escapes trajectory directory')
            encoded = _read(path)
            if _hash(encoded) != digest:
                raise ValueError('Trajectory payload hash mismatch')
            return self.decode(encoded, row, active)
        if kind in ('local', 'step'):
            if kind == 'step' and len(control) == 4:
                number, path, digest = control[1:]
                if type(number) is not int or number < 1:
                    raise ValueError('Invalid referenced step number')
                target_row = self._referenced_step(number)
            elif kind == 'local' and len(control) == 3:
                path, digest = control[1:]
                target_row = row
            else:
                raise ValueError('Invalid state reference')
            if not isinstance(path, list) or not path or path[0] not in ('state', 'action', 'metrics', 'reward', 'done', 'replay_context'):
                raise ValueError('Out of bounds trajectory reference')
            try:
                # Canonical fields may themselves be blob references. Expand
                # once before navigating paths indexed from the original data.
                target = self.decode(target_row[path[0]], target_row, active)
                for part in path[1:]:
                    if isinstance(target, list) and (type(part) is not int or part < 0):
                        raise ValueError('Invalid list reference')
                    target = target[part]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError('Missing trajectory reference target') from exc
            # The selected leaf is already decoded; interpreting it again
            # could turn escaped literal user JSON into a control envelope.
            if _hash(target) != digest:
                raise ValueError('Trajectory state reference hash mismatch')
            return target
        raise ValueError('Unknown trajectory reference')


def _rows(path, committed=False):
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            if committed and not line.endswith(b'\n'):
                return
            if line.strip():
                try:
                    yield json.loads(line.decode('utf-8-sig' if line_number == 1 else 'utf-8'))
                except ValueError as exc:
                    raise ValueError('Corrupt trajectory JSONL: %s, line %d' % (path, line_number)) from exc


class StepTraceStore:
    """Single-writer store. A new directory is required; resuming is read-only."""
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = {k: identity[k] for k in IDENTITY_KEYS if k in identity}
        self.codec = _Codec(self.directory)
        self.pending = []
        self.step = 0
        self.last_sequence = None
        self.last_row = None
        self._state_hashes = {}
        self.closed = False
        metadata = self.directory / 'episode.json'
        if metadata.exists():
            _identity(_read(metadata), self.identity)
        else:
            _atomic(metadata, dict(self.identity, schema_version=3,
                                  storage_features=['shared_state', 'row_event_defaults_v1']))
        steps = self.directory / 'steps.jsonl'
        if steps.exists() and steps.stat().st_size:
            raise ValueError('Cannot overwrite an existing step trajectory')
        if (self.directory / 'pending.json').exists():
            raise ValueError('Cannot overwrite pending trajectory evidence')
        # The recorder and repacker can precreate raw episode metadata. Only
        # update it after refusing existing committed/pending evidence.
        description = _read(metadata)
        features = description.get('storage_features', [])
        if not isinstance(features, list) or any(not isinstance(feature, str) for feature in features):
            raise ValueError('Invalid trajectory storage features')
        features = list(features)
        for feature in ('shared_state', 'row_event_defaults_v1'):
            if feature not in features:
                features.append(feature)
        if features != description.get('storage_features'):
            description['storage_features'] = features
            _atomic(metadata, description)
        steps.touch(exist_ok=True)

    def append(self, event):
        if self.closed:
            raise ValueError('Trajectory store is closed')
        event = json.loads(_bytes(event))
        _identity(event, self.identity)
        sequence = event.get('sequence')
        if sequence is not None:
            if type(sequence) is not int or (self.last_sequence is not None and sequence <= self.last_sequence):
                raise ValueError('Non-increasing trajectory sequence')
        if event.get('event_type', event.get('event')) == 'episode_start':
            if self.pending or self.step or self.last_sequence is not None:
                raise ValueError('Episode start must be the first event')
            metadata = _read(self.directory / 'episode.json')
            metadata['storage_schema_version'] = 3
            metadata['initial_event'] = self.codec.encode(event, root_inline=True)
            _atomic(self.directory / 'episode.json', metadata)
        else:
            pending = self.pending + [event]
            _atomic(self.directory / 'pending.json', dict(schema_version=3, identity=self.identity,
                    step=self.step + 1, events=[self.codec.encode(e, blobs=False) for e in pending]))
            self.pending = pending
        if sequence is not None:
            self.last_sequence = sequence

    def commit_step(self):
        steps = [e for e in self.pending if e.get('event_type', e.get('event')) == 'step']
        if not steps:
            return
        if len(steps) != 1:
            raise ValueError('Exactly one returned environment step is required')
        event = steps[0]
        number = self.step + 1
        _returned_steps(self.pending, number, required=True)
        row = dict(schema_version=3, step=number, identity=self.identity)
        row['events'] = []  # Explicit context for checked compact event defaults.
        local = {}
        for key in ('action', 'reward', 'done', 'metrics', 'state', 'replay_context'):
            if key in event:
                if key == 'state':
                    row[key], self._state_hashes = self.codec.encode_state(event[key], local, self._state_hashes)
                else:
                    row[key] = self.codec.encode(event[key], blobs=False)
                _index(event[key], [key], local)
        # Decision-boundary contexts stay in their original events, with the
        # latest effective snapshot also discoverable on the public step row.
        if 'replay_context' not in row:
            contexts = [e['replay_context'] for e in self.pending if 'replay_context' in e]
            if contexts:
                row['replay_context'] = self.codec.encode(contexts[-1])
                _index(contexts[-1], ['replay_context'], local)
        row['events'] = [self.codec.encode_event(e, local, row) for e in self.pending]
        with (self.directory / 'steps.jsonl').open('ab') as stream:
            stream.write(_bytes(row) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.step = number
        self.last_row = row
        self.pending = []
        (self.directory / 'pending.json').unlink(missing_ok=True)

    def write_summary(self, summary):
        _identity(summary, self.identity)
        local = {}
        if self.last_row:
            for key in ('state', 'action', 'metrics', 'replay_context'):
                if key in self.last_row:
                    _index(self.codec.decode(self.last_row[key], self.last_row), [key], local)
        encoded = self.codec.encode(summary, local, blobs=False)
        def external(value):
            if isinstance(value, dict):
                control = value.get(CONTROL)
                if isinstance(control, list) and control and control[0] == 'literal':
                    return {CONTROL: ['literal', {k: external(v) for k, v in control[1].items()}]}
                if isinstance(control, list) and control and control[0] == 'local':
                    return {CONTROL: ['step', self.step] + control[1:]}
                return {k: external(v) for k, v in value.items()}
            if isinstance(value, list):
                return [external(v) for v in value]
            return value
        _atomic(self.directory / 'summary.json', external(encoded))

    def close(self):
        self.closed = True


def iter_trace_events(path, *, include_pending=True):
    """Yield original expanded V1/V2 events or checked V3 step packets."""
    path = Path(path)
    if path.is_dir():
        path = path / ('steps.jsonl' if (path / 'steps.jsonl').exists() else 'events.jsonl')
    if path.name != 'steps.jsonl':
        yield from _rows(path)
        return
    directory = path.parent
    codec = _Codec(directory)
    metadata = _read(directory / 'episode.json')
    _identity(metadata, {})
    identity = {k: metadata[k] for k in IDENTITY_KEYS if k in metadata}
    last_sequence = None
    last_row_events = []
    count = 0
    def checked(event):
        nonlocal last_sequence
        if not isinstance(event, dict):
            raise ValueError('Trajectory event must be an object')
        _identity(event, identity)
        sequence = event.get('sequence')
        if sequence is not None:
            if type(sequence) is not int or (last_sequence is not None and sequence <= last_sequence):
                raise ValueError('Non-increasing trajectory sequence')
            last_sequence = sequence
        return event
    if 'initial_event' in metadata:
        yield checked(codec.decode(metadata['initial_event']))
    for row in _rows(path, True):
        count += 1
        _packet(row, 'step')
        if row.get('schema_version') != 3 or row.get('step') != count:
            raise ValueError('Non-contiguous or invalid step packet')
        _identity(row.get('identity', {}), identity)
        events = [codec.decode(e, row) for e in row['events']]
        step_events = _returned_steps(events, count, required=True)
        for key in ('action', 'reward', 'done', 'metrics', 'state'):
            if key in step_events[0] and (key not in row or codec.decode(row[key], row) != step_events[0][key]):
                raise ValueError('Step packet differs from original event')
        for event in events:
            yield checked(event)
        last_row_events = events
    pending_path = directory / 'pending.json'
    if include_pending and pending_path.exists():
        pending = _packet(_read(pending_path), 'pending')
        _identity(pending.get('identity', {}), identity)
        events = [codec.decode(e) for e in pending['events']]
        _returned_steps(events, pending['step'])
        if pending.get('step') == count and events == last_row_events:
            return  # A durable row won the race with pending cleanup.
        if pending.get('step') != count + 1:
            raise ValueError('Conflicting pending step evidence')
        for event in events:
            yield checked(event)


def read_trace_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'summary.json'
    elif path.suffix == '.jsonl':
        path = path.parent / 'summary.json'
    summary = _read(path)
    if (path.parent / 'steps.jsonl').exists():
        summary = _Codec(path.parent).decode(summary)
        _identity(summary, _read(path.parent / 'episode.json'))
    return summary
