"""demo: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from demo.task import Net, get_weights, set_weights, save_model

# Aggregation function! Don't worry about specifics...
class SaveModelFedAvg(FedAvg):
    def __init__(self, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None and rnd == self.num_rounds:
            print(f"Final round reached (round {rnd}). Saving global model...")
            net = Net()
            # Handle the possibility that aggregated_parameters is wrapped in a tuple.
            if isinstance(aggregated_parameters, tuple):
                params = aggregated_parameters[0]
            else:
                params = aggregated_parameters

            # Convert the Parameters object to a list of NumPy arrays.
            params_list = parameters_to_ndarrays(params)
            set_weights(net, params_list)
            model_save_path = "final_global_model.pth"
            save_model(net, file_path=model_save_path)
            print(f"Saved global model at {model_save_path}!")
        return aggregated_parameters

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define your custom strategy that saves the model after the final round
    strategy = SaveModelFedAvg(
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)