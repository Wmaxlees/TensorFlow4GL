from tensorflowdsl import runner

my_runner = runner.Runner()
my_runner.load_hyperparameters_from_json('./test.hyp')
my_runner.load_from_file('./test.mdl')
