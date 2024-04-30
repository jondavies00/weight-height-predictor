from .predictor import guess_user_input, classify_data, plot_error

results = classify_data()
plot_error(results.errors)
guess_user_input(results.thetas)
