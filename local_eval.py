from neurips2022nmmo import CompetitionConfig, scripted, submission, RollOut

config = CompetitionConfig()

my_team = submission.get_team_from_submission(
    submission_path="my-submission/",
    team_id="MyTeam",
    env_config=config,
)
# Or initialize the team directly
# my_team = MyTeam("Myteam", config, ...)

teams = [scripted.CombatTeam(f"Combat-{i}", config) for i in range(5)]
teams.extend([scripted.MixtureTeam(f"Mixture-{i}", config) for i in range(10)])
teams.append(my_team)

ro = RollOut(config, teams, parallel=True, show_progress=True)
ro.run(n_episode=1)
