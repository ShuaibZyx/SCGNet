import datetime
import randomname


def generate_experiment_name(experiment):
    now = datetime.datetime.now().strftime("%m%d%H%M")
    experiment_name = f"{now}_{experiment}_{randomname.get_name()}"
    return experiment_name
