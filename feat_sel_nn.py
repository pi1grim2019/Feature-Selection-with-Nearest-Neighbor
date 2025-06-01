import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import time

# Load the dataset
def load_data(file_path: str):
    if file_path.lower() == "wine":
        data = load_wine()
        X = data.data
        y = data.target
    else:
        data = np.loadtxt(file_path)
        X = data[:, 1:]
        y = data[:, 0].astype(int)
    return X, y


def my_knn(train_X, train_y, test_x):
    min_dist = float('inf')
    prediction = None
    for i in range(len(train_X)):
        dist = np.linalg.norm(train_X[i] - test_x)  # Compute the Euclidean distance
        if dist < min_dist:
            min_dist = dist
            prediction = train_y[i]  # The label of the nearest training sample
    return prediction

# Leave-One-Out Cross Validation
def leave_one_out(X: np.ndarray, y: np.ndarray):
    n_samples = len(X)
    correct = 0

    for i in range(n_samples):
        test_x = X[i]  # Use the current sample as the test sample
        test_y = y[i]
        train_X = np.delete(X, i, axis=0)  # Use the remaining samples as the training set
        train_y = np.delete(y, i, axis=0)

        pred = my_knn(train_X, train_y, test_x)
        if pred == test_y:
            correct += 1
    return correct / n_samples  # Return the accuracy


def forward_selection(X: np.ndarray, y: np.ndarray):
    n_features = X.shape[1]
    selected = []  # Currently selected features
    best_subset = []  # The feature subset with the highest accuracy

    # Compute baseline with no features
    baseline_accuracy = leave_one_out(np.zeros((X.shape[0], 0)), y)
    print(f"\nBaseline accuracy (using no features): {baseline_accuracy * 100:.1f}%")

    print(f"\nEvaluating 1-Nearest Neighbor using all {n_features} features with \"leaving-one-out\" cross-validation...")
    full_accuracy = leave_one_out(X, y)  # Initial accuracy of all features
    best_accuracy = full_accuracy
    prev_round_accuracy = 0.0
    print(f"Accuracy: {full_accuracy * 100:.1f}%\n")
    print("Beginning search.\n")

    # In each round, select one new feature and add it to the current subset
    while True:
        best_local_idx = None
        best_local_accuracy = 0.0

        for i in range(n_features):
            if i not in selected:
                tmp_features = selected + [i]
                acc = leave_one_out(X[:, tmp_features], y)
                feature_str = ", ".join(str(x + 1) for x in tmp_features)
                print(f"        Using feature(s) [{feature_str}], the accuracy is {acc * 100:.1f}%")
                if acc > best_local_accuracy:
                    best_local_accuracy = acc
                    best_local_idx = i

        if best_local_idx is not None:
            selected.append(best_local_idx)  # Add the current best feature
            feature_str = ", ".join(str(i + 1) for i in selected)
            print(f"\nFeature set [{feature_str}] has the highest accuracy, which is {best_local_accuracy * 100:.1f}%\n")

            if prev_round_accuracy > 0 and best_local_accuracy < prev_round_accuracy and len(selected) < n_features:
                print(f"[Warning] Accuracy dropped — continuing search to escape potential local maximum.\n")

            if best_local_accuracy > best_accuracy:
                best_accuracy = best_local_accuracy
                best_subset = selected.copy()

            prev_round_accuracy = best_local_accuracy
        else:
            break

    return best_subset, best_accuracy


def backward_elimination(X: np.ndarray, y: np.ndarray):
    n_features = X.shape[1]
    selected = list(range(n_features))  # Initially include all features
    best_subset = selected.copy()

    print(f"\nEvaluating 1-Nearest Neighbor using all {n_features} features with \"leaving-one-out\" cross-validation...")
    best_accuracy = leave_one_out(X[:, selected], y)
    print(f"Accuracy: {best_accuracy * 100:.1f}%\n")
    print("Beginning search.\n")

    while len(selected) > 1:
        best_local_accuracy = 0.0
        feature_to_remove = None

        for i in selected:
            tmp_features = [f for f in selected if f != i]
            acc = leave_one_out(X[:, tmp_features], y)
            feature_str = ", ".join(str(x + 1) for x in tmp_features)
            print(f"        Using feature(s) [{feature_str}], the accuracy is {acc * 100:.1f}%")
            if acc > best_local_accuracy:
                best_local_accuracy = acc
                feature_to_remove = i

        if feature_to_remove is not None:
            selected.remove(feature_to_remove)
            feature_str = ", ".join(str(x + 1) for x in selected)
            print(f"\nFeature set [{feature_str}] has the highest accuracy, which is {best_local_accuracy * 100:.1f}%\n")

            if best_local_accuracy > best_accuracy:
                best_accuracy = best_local_accuracy
                best_subset = selected.copy()
            else:
                if len(selected) > 1:
                    print(f"[Warning] Accuracy dropped — continuing search to escape potential local maximum.\n")
        else:
            break

    return best_subset, best_accuracy


def main():
    print("Welcome to the Feature Selection Algorithm Simulator!")
    file_path = input("Enter the name of the dataset to test: ").strip()
    print("Type the number of the algorithm you want to run:")
    print("1 for Forward Selection")
    print("2 for Backward Elimination")
    choice = input("Your choice: ").strip()

    X, y = load_data(file_path)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\nThis dataset contains {X.shape[1]} features (excluding the the class attribute), with {X.shape[0]} instances.")

    start_time = time.time()

    if choice == '1':
        selected, acc = forward_selection(X, y)
    elif choice == '2':
        selected, acc = backward_elimination(X, y)
    else:
        print("Invalid input. Please try again.")
        return

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Search completed. The optimal feature subset is {[x+1 for x in selected]}, achieving an accuracy of {acc * 100:.1f}%")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
