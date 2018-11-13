from t_test_models import T_Test_Validation

from sklearn.datasets import load_boston, load_digits

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

def main():
    data = load_boston() 
    X, y = data["data"], data["target"]

    t_validator = T_Test_Validation(validator=KFold, scorer=mean_squared_error)
    alpha_range = np.arange(0.01, 10, 0.5)

    base_model = LinearRegression()
    new_model = Ridge()

    t_stat = t_validator.validation_curve(X, y, base_model, new_model, "alpha", alpha_range)

    plt.plot(alpha_range, t_stat)
    plt.xlabel('alpha')
    plt.ylabel('t-statistic')
    plt.show()

if __name__ == "__main__":
    main()