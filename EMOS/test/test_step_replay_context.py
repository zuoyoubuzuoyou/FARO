"""Decision memory reconstruction, without clients, simulator or transport."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import json
import unittest


def load_agent():
    path = Path(__file__).resolve().parents[1] / 'habitat-mas/habitat_mas/agents/crab_agent.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    events = []
    namespace = dict(List=List, Optional=Optional, Action=object, OpenAIModel=object,
                     deepcopy=deepcopy, json=json, current_action_id=lambda name: 'action-1',
                     record_event=lambda kind, **data: events.append((kind, data)))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['CrabAgent'], namespace, events


class ReplayContextTests(unittest.TestCase):
    def setUp(self):
        self.Agent, self.ns, self.events = load_agent()
        self.agent = self.Agent('robot_0', [])
        self.agent.initialized = True
        self.agent.start_act = True
        self.agent.task_description = 'task'
        self.agent.subtask_description = 'subtask'
        self.agent.robot_type = 'fetch'
        self.agent.planner_plan = {'steps': ['pick']}
        self.agent.llm_model = SimpleNamespace(
            client=object(), model='test-model', window_size=4, openai_tools=[{'name': 'wait'}],
            tool_calls_enable=True, planning_stage=False, code_execution=False,
            system_message={'role': 'system', 'content': 'system'}, token_usage=17,
            chat_history=[[{'role': 'assistant', 'tool_calls': [{'id': 'call-1'}]},
                           {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'pending'}]])
        self.Agent.message_pipe = {'robot_1': ['first', 'second'], 'robot_0': []}

    def test_detached_round_trip_preserves_runtime_objects(self):
        self.agent._recovery_actions = [{'name': 'wait', 'arguments': {}}]
        self.agent._recovery_feedback = ['retry']
        saved = self.agent.export_replay_context()
        client, model, skills = self.agent.llm_model.client, self.agent.llm_model, self.agent.actions
        self.agent.llm_model.chat_history.clear()
        self.Agent.message_pipe['robot_1'].reverse()
        self.agent.start_act = False
        self.agent._recovery_feedback.clear()
        self.agent.restore_replay_context(saved)
        self.assertEqual(self.agent.export_replay_context(), saved)
        self.assertIs(self.agent.llm_model, model)
        self.assertIs(model.client, client)
        self.assertIs(self.agent.actions, skills)
        self.assertEqual(model.chat_history[-1][-1]['content'], 'pending')
        saved['message_pipe']['robot_1'].clear()
        self.assertEqual(self.Agent.message_pipe['robot_1'], ['first', 'second'])

    def test_incompatible_configuration_rejected_before_mutation(self):
        saved = self.agent.export_replay_context()
        self.agent.llm_model.window_size = 8
        before = deepcopy(self.agent.llm_model.chat_history)
        with self.assertRaises(ValueError):
            self.agent.restore_replay_context(saved)
        self.assertEqual(self.agent.llm_model.chat_history, before)

    def test_decoding_configuration_round_trips_and_rejects_changed_seed(self):
        self.agent.llm_model.temperature = 0.0
        self.agent.llm_model.seed = 100
        saved = self.agent.export_replay_context()
        self.assertEqual(saved['model_config']['temperature'], 0.0)
        self.assertEqual(saved['model_config']['seed'], 100)
        self.agent.restore_replay_context(saved)
        self.agent.llm_model.seed = 101
        with self.assertRaises(ValueError):
            self.agent.restore_replay_context(saved)

    def test_legacy_missing_optional_controls_match_only_unset_controls(self):
        self.agent.llm_model.temperature = None
        self.agent.llm_model.seed = None
        saved = self.agent.export_replay_context()
        saved['model_config'].pop('temperature', None)
        saved['model_config'].pop('seed', None)
        self.agent.restore_replay_context(saved)
        self.agent.llm_model.seed = 0
        with self.assertRaises(ValueError):
            self.agent.restore_replay_context(saved)

    def test_malformed_memory_rejected_without_any_mutation(self):
        cases = [
            ('model_memory', 'chat_history', None),
            ('model_memory', 'chat_history', [{'role': 'user', 'content': 'ungrouped'}]),
            ('model_memory', 'chat_history', [[None]]),
            ('model_memory', 'system_message', 'system'),
            ('model_memory', 'token_usage', '17'),
            ('agent_memory', 'start_act', 'yes'),
            ('agent_memory', '_recovery_feedback', [None]),
            ('agent_memory', '_recovery_actions', [{'name': 'pick'}]),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                before = self.agent.export_replay_context()
                bad = deepcopy(before)
                bad[section][key] = value
                bad['message_pipe'] = {'robot_1': ['changed']}
                with self.assertRaises(ValueError):
                    self.agent.restore_replay_context(bad)
                self.assertEqual(self.agent.export_replay_context(), before)

    def test_absent_model_rejects_memory_without_partial_agent_mutation(self):
        self.agent.llm_model = None
        before = self.agent.export_replay_context()
        bad = deepcopy(before)
        bad['agent_memory']['start_act'] = False
        bad['model_memory']['chat_history'] = []
        bad['message_pipe'] = {'robot_1': ['changed']}
        with self.assertRaises(ValueError):
            self.agent.restore_replay_context(bad)
        self.assertEqual(self.agent.export_replay_context(), before)

    def test_missing_memory_field_rejected(self):
        before = self.agent.export_replay_context()
        bad = deepcopy(before)
        del bad['model_memory']['chat_history']
        with self.assertRaises(ValueError):
            self.agent.restore_replay_context(bad)
        self.assertEqual(self.agent.export_replay_context(), before)

    def test_live_and_serialized_sdk_messages_restore(self):
        try:
            from openai.types.chat import ChatCompletionMessage
        except ImportError:
            self.skipTest('OpenAI SDK unavailable in this interpreter')
        assistant = ChatCompletionMessage(role='assistant', content=None, tool_calls=[{
            'id': 'call-1', 'type': 'function',
            'function': {'name': 'wait', 'arguments': '{}'}}])
        self.agent.llm_model.chat_history = [[assistant, {
            'role': 'tool', 'tool_call_id': 'call-1', 'content': 'pending'}]]
        live = self.agent.export_replay_context()
        self.agent.restore_replay_context(live)
        serialized = json.loads(json.dumps(live, default=lambda obj: obj.model_dump()))
        self.agent.restore_replay_context(serialized)
        self.assertEqual(self.agent.llm_model.chat_history, serialized['model_memory']['chat_history'])

    def test_valid_absent_model_and_minimal_model(self):
        for model in (None, SimpleNamespace()):
            self.agent.llm_model = model
            saved = self.agent.export_replay_context()
            self.agent.restore_replay_context(saved)
            self.assertIs(self.agent.llm_model, model)
            self.assertEqual(self.agent.export_replay_context(), saved)

    def test_disabled_recording_does_not_export(self):
        self.ns['current_action_id'] = lambda name: None
        self.agent.export_replay_context = lambda: self.fail('snapshot while disabled')
        self.agent._record_action_output(None, 'identity')
        self.assertNotIn('replay_context', self.events[-1][1])

    def test_send_request_snapshot_after_queue_and_final_history(self):
        self.agent.llm_model.chat = lambda obs: ('send_request', {'target_agent': 'robot_1', 'request': 'help'})
        result = self.agent._chat('observation')
        self.assertEqual(result, {'name': 'wait', 'arguments': ['500']})
        context = self.events[-1][1]['replay_context']
        self.assertEqual(context['agent_name'], 'robot_0')
        self.assertEqual(context['message_pipe']['robot_1'][:2], ['first', 'second'])
        self.assertIn('help', context['message_pipe']['robot_1'][-1])
        self.assertEqual(context['model_memory']['chat_history'][-1][-1]['content'], 'pending')

    def test_actor_event_order_reconstructs_global_pipe(self):
        other = self.Agent('robot_1', [])
        other.llm_model = SimpleNamespace(**vars(self.agent.llm_model))
        self.agent.llm_model.chat = lambda obs: ('send_request', {
            'target_agent': 'robot_1', 'request': 'help'})
        other.llm_model.chat = lambda obs: ('wait', {})
        self.agent._chat('first actor')
        other._chat('second actor')
        decisions = [data for kind, data in self.events if kind == 'agent_action']
        self.assertEqual([data['agent_name'] for data in decisions], ['robot_0', 'robot_1'])
        first, second = [data['replay_context'] for data in decisions]
        self.assertEqual(len(first['message_pipe']['robot_1']), 3)
        self.assertEqual(second['message_pipe']['robot_1'], [])
        self.Agent.message_pipe = {}
        self.agent.restore_replay_context(first)
        self.assertEqual(len(self.Agent.message_pipe['robot_1']), 3)
        other.restore_replay_context(second)
        self.assertEqual(self.Agent.message_pipe['robot_1'], [])

    def test_forced_recovery_snapshots_without_model_call(self):
        self.agent.llm_model.chat = lambda obs: self.fail('model transport used')
        self.agent._recovery_actions = [{'name': 'pick', 'arguments': {'target': 'cup'}}]
        self.agent._recovery_feedback = ['retry safely']
        result = self.agent._chat('observation')
        self.assertEqual(result['arguments']['robot'], 'robot_0')
        context = self.events[-1][1]['replay_context']
        self.assertEqual(context['agent_memory']['_recovery_actions'], [])
        self.assertEqual(context['agent_memory']['_recovery_feedback'], [])
        self.assertIn('retry safely', context['model_memory']['chat_history'][-1][0]['content'])


if __name__ == '__main__':
    unittest.main()
