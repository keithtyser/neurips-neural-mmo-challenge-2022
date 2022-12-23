import argparse

import nmmo


def render_replay(filepath):
    replay = nmmo.Replay.load(filepath)
    replay.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="Replay file(XXX.lzma)")
    args = parser.parse_args()
    render_replay(args.filepath)
