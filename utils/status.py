# status.py
is_training = False
is_evaluating = False

rmse_all = 0
rmse_neighborhood = 0
rmse_matrix_factorization = 0
rmse_content_based = 0
mae_all = 0
mae_neighborhood = 0
mae_matrix_factorization = 0
mae_content_based = 0


def get_evaluation_results():
    """
    This function returns the evaluation results.
    """
    return (
        round(rmse_all, 2),
        round(rmse_neighborhood, 2),
        round(rmse_matrix_factorization, 2),
        round(rmse_content_based, 2),
        round(mae_all, 2),
        round(mae_neighborhood, 2),
        round(mae_matrix_factorization, 2),
        round(mae_content_based, 2),
    )


def api_ready():
    """
    This function checks if the API is ready to receive requests.
    """
    if is_training:
        print("Model is currently training")
        return "Model is currently training"
    elif is_evaluating:
        print("Model is currently evaluating")
        return "Model is currently evaluating"
    else:
        return "Model is ready"
