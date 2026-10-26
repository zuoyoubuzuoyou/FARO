"""Configured decoding reaches real request construction and trace evidence."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_llm_model_selection import RequestTransport
from trajectory_test_utils import read_events
from habitat_mas.utils import models
from habitat_mas.utils.trajectory import EvaluationTrace


class LLMDecodingControlTests(unittest.TestCase):
    def requests_and_events(self, configured, calls):
        transport = RequestTransport()
        client = SimpleNamespace(chat=SimpleNamespace(completions=transport))
        with tempfile.TemporaryDirectory() as output:
            with EvaluationTrace(output) as trace:
                trace.start_episode(0, SimpleNamespace(scene_id='scene', episode_id='decoding'))
                with patch.object(models.openai, 'OpenAI', return_value=client):
                    model = models.OpenAIModel('system', [], save_on_each_chat=False, **configured)
                with trace.policy_step():
                    for phase, kwargs in calls:
                        model._create_completion([{'role': 'user', 'content': phase}], phase, **kwargs)
                events = read_events(trace.run_dir)
        return transport.requests, events

    def test_configured_zero_values_reach_all_phases_and_explicit_trace_sources(self):
        phases = ['discussion', 'planning', 'execution']
        requests, events = self.requests_and_events(
            {'temperature': 0.0, 'seed': 0}, [(phase, {}) for phase in phases])
        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertEqual(request['temperature'], 0.0)
            self.assertEqual(request['seed'], 0)
        configs = [event for event in events if event['event_type'] == 'model_config']
        evidence = [event for event in events if event['event_type'] == 'llm_request']
        self.assertEqual([event['phase'] for event in configs], phases)
        self.assertEqual([event['phase'] for event in evidence], phases)
        for config, event, request in zip(configs, evidence, requests):
            self.assertEqual(config['decoding_parameters']['temperature'], 0.0)
            self.assertEqual(config['decoding_parameters']['seed'], 0)
            for name in ('temperature', 'seed'):
                self.assertEqual(config['parameter_sources'][name], 'explicit_request')
                self.assertEqual(config['request_kwargs'][name], request[name])
                self.assertEqual(event['request_kwargs'][name], request[name])
                self.assertEqual(event[name], request[name])

    def test_per_call_controls_override_configured_values(self):
        requests, events = self.requests_and_events(
            {'temperature': 0.5, 'seed': 10},
            [('planning', {'temperature': 0.2, 'seed': 42}), ('execution', {})])
        self.assertEqual([(r['temperature'], r['seed']) for r in requests], [(0.2, 42), (0.5, 10)])
        evidence = [e for e in events if e['event_type'] == 'llm_request']
        self.assertEqual([(e['temperature'], e['seed']) for e in evidence], [(0.2, 42), (0.5, 10)])

    def test_unset_controls_are_omitted_from_request_and_marked_provider_defaults(self):
        requests, events = self.requests_and_events({}, [('discussion', {})])
        config = next(e for e in events if e['event_type'] == 'model_config')
        evidence = next(e for e in events if e['event_type'] == 'llm_request')
        for name in ('temperature', 'seed'):
            self.assertNotIn(name, requests[0])
            self.assertNotIn(name, evidence['request_kwargs'])
            self.assertNotIn(name, config['request_kwargs'])
            self.assertEqual(config['parameter_sources'][name], 'provider_default_not_sent')

    def test_legacy_instance_without_control_attributes_omits_them(self):
        transport = RequestTransport()
        client = SimpleNamespace(chat=SimpleNamespace(completions=transport))
        with patch.object(models.openai, 'OpenAI', return_value=client):
            model = models.OpenAIModel('system', [], save_on_each_chat=False)
        del model.temperature
        del model.seed
        model._create_completion([{'role': 'user', 'content': 'legacy instance'}], 'planning')
        self.assertEqual(len(transport.requests), 1)
        self.assertNotIn('temperature', transport.requests[0])
        self.assertNotIn('seed', transport.requests[0])
