"""Start an isolated local browser-test server with measured demo evaluation reports."""
import os
import uvicorn
from scripts.seed_evaluation_reports import run

if __name__=='__main__':
    if not os.getenv('MAINTENANCEAI_DB') or 'e2e-' not in os.environ['MAINTENANCEAI_DB']:
        raise RuntimeError('Browser verification requires its own explicit e2e database')
    run()
    uvicorn.run('backend.main:app',host='127.0.0.1',port=8000)
