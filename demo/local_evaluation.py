from pathlib import Path

from neurips2022nmmo import CompetitionConfig, RollOut, scripted, submission


def evaluate(submission_path: str):
    config = CompetitionConfig()
    config.SAVE_REPLAY = "eval"
    # uncomment the code below if you want to render locally
    # config.RENDER = True

    my_team = submission.get_team_from_submission(
        submission_path=submission_path,
        team_id="MyTeam",
        env_config=config,
    )
    # or initialize the team directly
    # my_team = MyTeam("Myteam", config, ...)

    teams = [scripted.CombatTeam(f"Combat-{i}", config) for i in range(5)]
    teams.extend(
        [scripted.MixtureTeam(f"Mixture-{i}", config) for i in range(10)])
    teams.append(my_team)

    ro = RollOut(config, teams, parallel=True, show_progress=True)
    ro.run(n_episode=1, render=config.RENDER)


if __name__ == "__main__":
    submission_path = Path(__file__).parent.parent.joinpath(
        "my-submission").resolve().as_posix()
    evaluate(submission_path)
