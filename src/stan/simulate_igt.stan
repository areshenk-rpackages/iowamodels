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
  
  array[NUM_TRIALS] vector[NUM_DECKS] win;
  array[NUM_TRIALS] vector[NUM_DECKS] loss;
}

parameters {
  array[NUM_UTILITY_PARAMETERS] real<lower=0, upper=1> utility_params;
  array[NUM_UPDATING_PARAMETERS] real<lower=0, upper=1> updating_params;
  array[NUM_TEMPERATURE_PARAMETERS] real<lower=0, upper=1> temperature_params;
}

generated quantities {
    array[NUM_TRIALS] vector[NUM_DECKS] V;
    array[NUM_TRIALS] simplex[NUM_DECKS] P;
    vector[NUM_TRIALS] wins;
    vector[NUM_TRIALS] losses;
    vector[NUM_TRIALS] U;
    vector[NUM_DECKS] n_choices;
    real theta;
    vector[NUM_DECKS] V_dummy;
    array[NUM_DECKS] int card;
    array[NUM_TRIALS] int choice;
    for (d in 1:NUM_DECKS){
        V_dummy[d] = 0;
        card[d] = 0;
    }

    // Play IGT
    for (t in 1:NUM_TRIALS){

        // Deck values for each trial
        V[t] = V_dummy;

        // Compute temperature
        theta = temperature(t, temperature_params, TEMPERATURE_FUNCTION);

        // Draw card
        P[t] = softmax(theta * V[t]);
        choice[t] = categorical_rng(P[t]);
        card[choice[t]] = card[choice[t]] + 1;

        // Wins and losses
        wins[t]  = win[card[choice[t]], choice[t]];
        losses[t] = loss[card[choice[t]], choice[t]];

        // Compute utility
        U[t] = utility(wins[t], losses[t],
                       utility_params, UTILITY_FUNCTION);

        // Update deck values
        V_dummy = updating(V_dummy, U[t], choice[t],
                           updating_params, UPDATING_FUNCTION);
    }
}
