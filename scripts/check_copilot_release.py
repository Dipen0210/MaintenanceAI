"""Failing exit status prevents CI release when measured evidence or expert review is missing."""
import argparse,json
from pathlib import Path
from backend.copilot.evaluation import release_reasons


def main():
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--review',type=Path)
    parser.add_argument('--technical-only',action='store_true',help='Check software gates only; this is not release approval')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text());baseline=json.loads(args.baseline.read_text()) if args.baseline else None
    review=json.loads(args.review.read_text()) if args.review else None
    if review and (review.get('run_id')!=report['id'] or not review.get('qualification') or not review.get('reviewer') or set(review.get('reviewed_case_ids',[]))!={c['id'] for c in report['cases']}):
        raise SystemExit('Expert review must identify this run, reviewer, qualification and every reviewed example')
    reasons=release_reasons(report,baseline,require_expert=not args.technical_only,expert_review=review)
    result={'run_id':report['id'],'configuration_id':report['configuration']['id'],'scope':'technical checks only, not release approval' if args.technical_only else 'release gate',
            'status':'blocked' if reasons else 'passed','reasons':reasons,'review':review}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if reasons:raise SystemExit(1)


if __name__=='__main__':main()
