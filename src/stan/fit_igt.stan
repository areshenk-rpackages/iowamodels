functions {
#include /include/utilityFunctions.stanfunctions
#include /include/updatingFunctions.stanfunctions
#include /include/temperatureFunctions.stanfunctions

#include /include/utilityWrapper.stanfunctions
#include /include/updatingWrapper.stanfunctions
#include /include/temperatureWrapper.stanfunctions
}

data {
  int<lower=0> NUM_TRIALS;
  int<lower=0> NUM_DECKS;

  int UTILITY_FUNCTION;
  int UPDATING_FUNCTION;
  int TEMPERATURE_FUNCTION;

  int<lower=0> NUM_UTILITY_PARAMETERS;
  int<lower=0> NUM_UPDATING_PARAMETERS;
  int<lower=0> NUM_TEMPERATURE_PARAMETERS;

  array[NUM_UTILITY_PARAMETERS] real UTILITY_LOWER_BOUND;
  array[NUM_UPDATING_PARAMETERS] real UPDATING_LOWER_BOUND;
  array[NUM_TEMPERATURE_PARAMETERS] real TEMPERATURE_LOWER_BOUND;

  array[NUM_UTILITY_PARAMETERS] real UTILITY_UPPER_BOUND;
  array[NUM_UPDATING_PARAMETERS] real UPDATING_UPPER_BOUND;
  array[NUM_TEMPERATURE_PARAMETERS] real TEMPERATURE_UPPER_BOUND;
  
  array[NUM_TRIALS] real win;
  array[NUM_TRIALS] real loss;
  array[NUM_TRIALS] int choice;

  real<lower=1> reg;
}

parameters {
  array[NUM_UTILITY_PARAMETERS] real<lower=0, upper=1> raw_utility_params;
  array[NUM_UPDATING_PARAMETERS] real<lower=0, upper=1> raw_updating_params;
  array[NUM_TEMPERATURE_PARAMETERS] real<lower=0, upper=1> raw_temperature_params;
}

transformed parameters {
  array[NUM_UTILITY_PARAMETERS] real<lower=0, upper=1> utility_params;
  array[NUM_UPDATING_PARAMETERS] real<lower=0, upper=1> updating_params;
  array[NUM_TEMPERATURE_PARAMETERS] real<lower=0, upper=1> temperature_params;

  for (i in 1:NUM_UTILITY_PARAMETERS){
      utility_params[i] = (UTILITY_UPPER_BOUND[i] - UTILITY_LOWER_BOUND[i]) *
                          raw_utility_params[i] + UTILITY_LOWER_BOUND[i];
  }

  for (i in 1:NUM_UPDATING_PARAMETERS){
      updating_params[i] = (UPDATING_UPPER_BOUND[i] - UPDATING_LOWER_BOUND[i]) *
                          raw_updating_params[i] + UPDATING_LOWER_BOUND[i];
  }

  for (i in 1:NUM_TEMPERATURE_PARAMETERS){
      temperature_params[i] = (TEMPERATURE_UPPER_BOUND[i] -
                               TEMPERATURE_LOWER_BOUND[i]) *
                          raw_temperature_params[i] + TEMPERATURE_LOWER_BOUND[i];
  }

}

model {

    // Dummy variables
    vector[NUM_DECKS]  V;
    real theta;
    real U;
    for (d in 1:4){
        V[d] = 0;
    }

    // Likelihood
    for (t in 1:NUM_TRIALS){

        // Compute temperature
        theta = temperature(t, temperature_params, TEMPERATURE_FUNCTION);

        // Draw card
        choice[t] ~ categorical_logit(theta * V);

        // Compute utility
        U = utility(win[t], loss[t], utility_params, UTILITY_FUNCTION);

        // Update deck values
        V = updating(V, U, choice[t], updating_params, UPDATING_FUNCTION);
    }

    // Priors
    raw_utility_params ~ beta(reg, reg);
    raw_updating_params ~ beta(reg, reg);
    raw_temperature_params ~ beta(reg, reg);
}
