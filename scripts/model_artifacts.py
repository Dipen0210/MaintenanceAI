"""Checksum-verified local model bundles. No implicit remote downloads or pickle inspection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/manifest.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory, manifest=None):
    manifest = manifest or json.loads(MANIFEST.read_text())
    for entry in manifest['artifacts']:
        path = Path(directory) / entry['filename']
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f"Missing or checksum mismatch: {entry['filename']}")
    return manifest


def export_bundle(directory, destination):
    manifest = verify(directory)
    with tarfile.open(destination, 'w:gz') as archive:
        for entry in manifest['artifacts']:
            archive.add(Path(directory)/entry['filename'], arcname=entry['filename'])


def import_bundle(source, directory):
    manifest = json.loads(MANIFEST.read_text())
    expected = {entry['filename']:entry['sha256'] for entry in manifest['artifacts']}
    payloads = {}
    with tarfile.open(source, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Unexpected or missing archive members')
        for member in members:
            if not member.isfile() or member.size > 100 * 1024 * 1024:
                raise ValueError('Invalid model archive member')
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError(f'Checksum mismatch: {member.name}')
            payloads[member.name] = data
    # Validate the whole bundle before installing any files; never extract paths.
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for name, data in payloads.items():
        temp = directory / (name+'.tmp')
        temp.write_bytes(data); temp.replace(directory/name)
    verify(directory)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['verify','export','import'])
    parser.add_argument('--directory',default=str(ROOT/'trained_models'))
    parser.add_argument('--bundle')
    args=parser.parse_args()
    if args.action=='verify':verify(args.directory)
    elif args.action=='export':export_bundle(args.directory,args.bundle)
    else:import_bundle(args.bundle,args.directory)
    print('Artifact operation verified:',args.action)

if __name__=='__main__':main()
