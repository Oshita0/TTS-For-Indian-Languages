import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch


ASSETS = Path('pilot_assets')
PILOT = ASSETS / 'results_mps4'
PANEL = ASSETS / 'panel'
OUT = ASSETS / 'results_lora_only'
CHECKPOINTS = {
    'adapter': ASSETS / 'emotion_v2.pt',
    'duration': ASSETS / 'duration_download.pt',
    'backbone': ASSETS / 'sooktam2' / 'sooktam.safetensors',
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def generate(cases, settings):
    from sooktam_rasa import infer
    from sooktam_rasa.emotion_dit import LoRALinear

    for name, path in CHECKPOINTS.items():
        if file_hash(path) != settings['checkpoints'][name]:
            raise ValueError(f'{name} checkpoint differs from the original pilot')

    model = infer.EmotionalSooktam.load(str(CHECKPOINTS['backbone'].parent),
                                        adapter=str(CHECKPOINTS['adapter']),
                                        duration_model=str(CHECKPOINTS['duration']),
                                        device=settings['device'])
    transformer = model.cfm.transformer
    layers = [module for module in transformer.modules() if isinstance(module, LoRALinear)]
    if not layers:
        raise ValueError('No LoRA layers found')
    original_sampler = infer.sample_with_emotion
    set_lora = infer.set_lora_enabled

    def verify_and_sample(cfm, mel, tokens, duration, emotion_weights, **kwargs):
        if emotion_weights is not None or kwargs['cfg_emo'] != 0 or not all(layer.enabled for layer in layers):
            raise ValueError('The requested LoRA-only condition is not active')
        return original_sampler(cfm, mel, tokens, duration, emotion_weights, **kwargs)

    # Keep LoRA active while the original sampler follows its unconditioned branch.
    with patch.object(infer, 'set_lora_enabled', side_effect=lambda module, _: set_lora(module, True)), \
         patch.object(infer, 'sample_with_emotion', side_effect=verify_and_sample):
        for speaker, base, _ in cases:
            path = OUT / f"{speaker['id']}-lora-only.wav"
            base_info = sf.info(PILOT / base['path'])
            if path.is_file():
                existing = sf.info(path)
                if (existing.frames, existing.samplerate) != (base_info.frames, base_info.samplerate):
                    raise ValueError(f'Existing clip has a different duration: {path}')
                continue
            ref = speaker['ref']
            print(f"Generating LoRA-only clip for {speaker['id']}", flush=True)
            wave, rate = model.synthesize(str(PANEL / ref['path']), ref['text'], base['text'], base['lang'],
                                         rasa=None, cfg_emo=0, nfe_step=settings['steps'],
                                         duration='learned', seed=settings['seed'])
            if (len(wave), rate) != (base_info.frames, base_info.samplerate):
                raise ValueError(f"Generated duration differs from base for {speaker['id']}")
            if not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-5:
                raise ValueError(f"Invalid waveform for {speaker['id']}")
            temp = path.with_suffix('.partial.wav')
            sf.write(temp, wave, rate)
            temp.replace(path)
            print(f"Saved {speaker['id']} with the original duration", flush=True)


def score(cases):
    from sooktam_rasa.evaluate import SpeakerSim

    scorer = SpeakerSim(device='cpu')
    rows = []
    for speaker, base, shanta in cases:
        speaker_id = speaker['id']
        enroll = str(PANEL / speaker['enrollment']['path'])
        with torch.inference_mode():
            lora = scorer(str(OUT / f'{speaker_id}-lora-only.wav'), enroll)
        original = float(base['speaker_similarity'])
        rows.append(dict(speaker=speaker_id, language=speaker['lang'], base=original, lora_only=lora,
                         shanta=float(shanta['speaker_similarity']), lora_minus_base=lora - original))
    path = OUT / 'paired_speaker_scores.csv'
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main():
    settings = json.loads((PILOT / 'settings.json').read_text())
    panel_path = PANEL / 'panel.json'
    if file_hash(panel_path) != settings['panel_sha256'] or settings['sentences'] != 1:
        raise ValueError('Panel or sentence count differs from the original pilot')
    panel = json.loads(panel_path.read_text())
    with (PILOT / 'rows_scored_speaker.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    cases = []
    for speaker in panel['speakers']:
        [base] = [row for row in rows if row['id'] == speaker['id'] and row['condition'] == 'base']
        [shanta] = [row for row in rows if row['id'] == speaker['id'] and row['condition'] == 'v2_cfg1'
                    and row['target'] == 'shanta']
        if base['text'] != shanta['text'] or base['audio_seconds'] != shanta['audio_seconds']:
            raise ValueError('The original base and neutral clips do not match in text and duration')
        cases.append((speaker, base, shanta))
    if len(cases) != 4:
        raise ValueError('Expected the original four speakers')
    if sys.argv[1:] == ['generate']:
        OUT.mkdir(exist_ok=True)
        generate(cases, settings)
    elif sys.argv[1:] == ['score']:
        score(cases)
    else:
        raise SystemExit('Pass generate or score')


if __name__ == '__main__':
    main()
