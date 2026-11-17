"""How the distillation inputs for the QR names model were made.

  uv run python names/teacher.py teacher   # names/teacher.pt: float GRU, embedding 16, hidden 96
  uv run python names/teacher.py samples   # names/teacher_samples.json: 20,000 draws, filtered
  uv run python names/teacher.py student   # names/student_real_plus_20k_teacher_samples.pt

`names/qrtf.py train` reads teacher.pt and teacher_samples.json for its soft
targets, and the published MLP was fine-tuned from the student checkpoint
(`--init`). This is a condensed copy of the original teacher.py and student.py
(see git history); their unrelated side experiments are left out. Python's
global RNG was never seeded in train2.train, so reruns will not be bit-identical.
"""
import argparse
import json

from train2 import Net, sample, torch, train, val


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['teacher', 'samples', 'student'])
    step = p.parse_args().step
    if step == 'teacher':
        m, _ = train('gru', 16, 96, 0, steps=5000, drop=0.3, wd=5e-2)
        torch.save(m.state_dict(), 'names/teacher.pt')
    elif step == 'samples':
        T = Net('gru', 16, 96, 0, 3, 0.3)
        T.load_state_dict(torch.load('names/teacher.pt'))
        torch.manual_seed(5)
        big = [s for s in sample(T, 20000, temp=1.0, maxlen=30) if 4 <= len(s) <= 24 and s not in set(val)]
        print('teacher samples', len(big))
        json.dump(big, open('names/teacher_samples.json', 'w'))
    else:
        big = json.load(open('names/teacher_samples.json'))
        m, v = train('mlp', 6, 24, 30, k=3, pos=1, extra=big, extra_w=0.7, steps=16000)
        torch.save(m.state_dict(), 'names/student_real_plus_20k_teacher_samples.pt')


if __name__ == '__main__':
    main()
