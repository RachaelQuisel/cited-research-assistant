import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'research.py'
spec = importlib.util.spec_from_file_location('research', SCRIPT)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


class RetrievalTests(unittest.TestCase):
    def test_arxiv_preserves_version_dates_and_access_scope(self):
        feed = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/1706.03762v7</id><title>Attention Is All You Need</title>
        <published>2017-06-12T00:00:00Z</published><updated>2023-08-02T00:00:00Z</updated>
        <author><name>Ashish Vaswani</name></author><summary>The model uses attention.</summary>
        </entry></feed>'''
        with patch.object(research, 'request', return_value=feed):
            result = research.search_arxiv('attention', '2017-01-01', '2017-12-31')
        source = result['sources'][0]
        self.assertEqual(source['identifier'], '1706.03762v7')
        self.assertEqual(source['submitted_date'], '2017-06-12T00:00:00Z')
        self.assertEqual(source['updated_date'], '2023-08-02T00:00:00Z')
        self.assertIsNone(source['publication_date'])
        self.assertEqual(source['scope'], 'abstract')
        self.assertEqual(source['publication_status'], 'preprint')
        self.assertEqual(source['content'], 'The model uses attention.')

    def test_semantic_scholar_absent_abstract_is_metadata_only(self):
        response = {'data':[{'paperId':'abc', 'title':'A paper', 'url':'https://www.semanticscholar.org/paper/abc',
                             'abstract':None, 'authors':[], 'externalIds':{'ArXiv':'1706.03762'}, 'publicationDate':None}]}
        with patch.object(research, 'request', return_value=json.dumps(response)):
            source = research.search_scholar('attention')['sources'][0]
        self.assertEqual(source['scope'], 'metadata')
        self.assertEqual(source['content'], 'A paper')
        self.assertIsNone(source['publication_date'])
        self.assertEqual(source['publication_status'], 'unknown')

    def test_unavailable_source_never_becomes_empty_success(self):
        with patch.object(research, 'request', side_effect=OSError('HTTP 429')):
            result = research.search_scholar('attention')
        self.assertEqual(result['sources'], [])
        self.assertIn('429', result['unavailable'])
        self.assertIn('WebSearch', result['fallback'])

    def test_date_filter_rejects_unknown_publication_date(self):
        response = {'data':[{'paperId':'abc','title':'A paper','abstract':None,'externalIds':{}}]}
        with patch.object(research, 'request', return_value=json.dumps(response)):
            result = research.search_scholar('attention', '2017-01-01', '2017-12-31')
        self.assertEqual(result['sources'], [])
        self.assertEqual(result['unknown_date_count'], 1)

    def test_duplicates_identify_same_work_without_losing_versions(self):
        sources = [{'source_id':'a', 'external_ids':{'arxiv':'1706.03762v1'}},
                   {'source_id':'b', 'external_ids':{'arxiv':'1706.03762v7'}},
                   {'source_id':'c', 'external_ids':{'arxiv':'2001.00001v1'}}]
        self.assertEqual(research.duplicate_groups(sources), [['a','b']])
        self.assertEqual(sources[1]['external_ids']['arxiv'], '1706.03762v7')

    def test_cross_provider_identifier_capitalization_does_not_hide_duplicates(self):
        sources = [{'source_id':'a', 'external_ids':{'arxiv':'1706.03762v1'}},
                   {'source_id':'b', 'external_ids':{'ArXiv':'1706.03762v7'}}]
        self.assertEqual(research.duplicate_groups(sources), [['a','b']])


class ClaimTests(unittest.TestCase):
    def records(self):
        return {'sources':[{'source_id':'s1', 'identifier':'fixture-v1', 'title':'Fictional study',
                            'url':'https://example.org/study', 'scope':'abstract',
                            'content':'The fixture sample contained 12 participants.',
                            'publication_status':'preprint','retrieved_at':'2026-10-03T00:00:00Z'}],
                'claims':[{'claim_id':'c1', 'text':'The fixture sample contained 12 participants.',
                           'status':'supported', 'kind':'substantive', 'inference':False,
                           'support_review':'The abstract explicitly states this sample size.',
                           'evidence':[{'source_id':'s1','passage':'sample contained 12 participants',
                                        'relationship':'supports'}]}]}

    def test_cited_output_exposes_scope_preprint_and_supporting_passage(self):
        output = research.render(self.records())
        self.assertIn('[s1:', output)
        self.assertIn('https://example.org/study', output)
        self.assertIn('abstract', output)
        self.assertIn('preprint', output)
        self.assertIn('sample contained 12 participants', output)

    def test_fabricated_reference_cannot_become_supported(self):
        records = self.records()
        records['claims'][0]['evidence'][0]['source_id'] = 'nonexistent'
        with self.assertRaisesRegex(ValueError, 'source'):
            research.render(records)

    def test_real_reference_with_absent_passage_is_rejected(self):
        records = self.records()
        records['claims'][0]['evidence'][0]['passage'] = 'The model cures cancer.'
        with self.assertRaisesRegex(ValueError, 'passage'):
            research.render(records)

    def test_metadata_cannot_support_a_substantive_claim(self):
        records = self.records()
        records['sources'][0]['scope'] = 'metadata'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            research.render(records)

    def test_absent_abstract_detail_and_missing_paper_are_declined(self):
        records = self.records()
        records['claims'] = [{'claim_id':'c2','text':'The fixture used randomized allocation.',
                              'status':'unverifiable','reason':'The abstract does not report allocation.', 'evidence':[]},
                             {'claim_id':'c3','text':'A nonexistent paper confirms the finding.',
                              'status':'unverifiable','reason':'No matching paper was retrieved.', 'evidence':[]}]
        output = research.render(records)
        self.assertEqual(output.count("I can't verify this"), 2)
        self.assertNotIn('## Verified claims\n- ', output)

    def test_gap_explanation_has_its_own_source_citation(self):
        records = self.records()
        records['claims'] = [{'claim_id':'c2','text':'The fixture randomized allocation.',
                              'status':'unverifiable','reason':'The available abstract reports the sample only.',
                              'reason_source_ids':['s1'],'evidence':[]}]
        output = research.render(records)
        gap_line = next(line for line in output.splitlines() if "I can't verify this" in line)
        self.assertIn('[s1:', gap_line)
        records['claims'][0]['reason_source_ids'] = ['nonexistent']
        with self.assertRaisesRegex(ValueError, 'source'):
            research.render(records)

    def test_inference_is_explicit_and_disagreement_cannot_be_hidden(self):
        records = self.records()
        records['claims'][0]['inference'] = True
        self.assertIn('Inference:', research.render(records))
        contrary = dict(records['sources'][0], source_id='s2', title='Fictional conflicting study',
                        identifier='conflicting-fixture-v1', url='https://example.org/other-study',
                        content='The fixture sample contained 14 participants.')
        records['sources'].append(contrary)
        records['claims'][0]['evidence'].append({'source_id':'s2','passage':'14 participants','relationship':'contradicts'})
        with self.assertRaisesRegex(ValueError, 'disagreement'):
            research.render(records)
        records['claims'][0]['status'] = 'unverifiable'
        records['claims'][0]['reason'] = 'Retrieved sources disagree.'
        output = research.render(records)
        self.assertIn('contradicts', output)
        self.assertIn("I can't verify this", output)

    def test_final_hook_detects_actual_uncited_addition_then_accepts_exact_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            research.begin(data, 'session-test')
            expected = research.prepare(data, 'session-test', self.records())
            result = research.final_check(data, {'session_id':'session-test','last_assistant_message':expected+'\nThe model is always accurate.'})
            self.assertEqual(result['decision'], 'block')
            result = research.final_check(data, {'session_id':'session-test','stop_hook_active':True,'last_assistant_message':expected})
            self.assertEqual(result, {})
            audits = [json.loads(s) for s in (data/'audits/session-test.jsonl').read_text().splitlines()]
            self.assertEqual([a['coverage_pass'] for a in audits], [False,True])
            self.assertEqual(research.final_check(data, {'session_id':'session-test','last_assistant_message':'An unrelated later reply.'}), {})

    def test_repeated_bad_response_stops_with_failure_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            research.begin(data, 'session-test')
            result = research.final_check(data, {'session_id':'session-test','stop_hook_active':True,'last_assistant_message':'Uncited.'})
            self.assertFalse(result['continue'])
            self.assertIn('failed', result['stopReason'])
            self.assertFalse(json.loads((data/'sessions/session-test.json').read_text())['completed'])


if __name__ == '__main__':
    unittest.main()
