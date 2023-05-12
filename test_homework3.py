from pages import A_Explore_Preprocess_Data, B_Train_Model, C_Test_Model
import pandas as pd
import numpy as np
import string

############## Assignment 3 Inputs #########
student_filepath = "./datasets/Amazon Product Reviews I.csv"
grader_filepath = "./test_files/Amazon Product Reviews I.csv"
student_df = pd.read_csv(student_filepath)


# Checkpoint 1
def contains_punctuation(s):
    if isinstance(s, str):
        return any(c in string.punctuation for c in s)
    else:
        return False


def assert_no_punctuation(df, columns):
    for col in columns:
        assert not df[col].apply(contains_punctuation).any()


def test_remove_punctuation():
    student_df_copy = pd.read_csv(student_filepath)
    std_cl_df, _ = A_Explore_Preprocess_Data.clean_data(student_df_copy)
    std_rm_punc_df = A_Explore_Preprocess_Data.remove_punctuation(
        std_cl_df, ['reviews', 'title'])
    assert_no_punctuation(std_rm_punc_df, ['reviews', 'title'])


# Checkpoint 2
def test_tf_idf_encoder():
    student_df_copy = pd.read_csv(student_filepath)
    std_cl_df, _ = A_Explore_Preprocess_Data.clean_data(student_df_copy)
    std_rm_punc_df = A_Explore_Preprocess_Data.remove_punctuation(
        std_cl_df, ['reviews', 'title'])
    std_review_df = A_Explore_Preprocess_Data.tf_idf_encoder(
        std_rm_punc_df, 'reviews', [])
    std_title_df = A_Explore_Preprocess_Data.tf_idf_encoder(
        std_rm_punc_df, 'title', [])
    expected_review_df = pd.read_pickle(
        "./test_files/tf_idf_encoder_review.pkl")
    expected_title_df = pd.read_pickle("./test_files/tf_idf_encoder_title.pkl")
    pd.testing.assert_frame_equal(std_review_df, expected_review_df)
    pd.testing.assert_frame_equal(std_title_df, expected_title_df)


# Checkpoint 3
def test_word_count_encoder():
    student_df_copy = pd.read_csv(student_filepath)
    std_cl_df, _ = A_Explore_Preprocess_Data.clean_data(student_df_copy)
    std_rm_punc_df = A_Explore_Preprocess_Data.remove_punctuation(
        std_cl_df, ['reviews', 'title'])
    std_review_df = A_Explore_Preprocess_Data.word_count_encoder(
        std_rm_punc_df, 'reviews', [])
    std_title_df = A_Explore_Preprocess_Data.word_count_encoder(
        std_rm_punc_df, 'title', [])
    expected_review_df = pd.read_pickle(
        "./test_files/word_count_encoder_review.pkl")
    expected_title_df = pd.read_pickle(
        "./test_files/word_count_encoder_title.pkl")
    pd.testing.assert_frame_equal(std_review_df, expected_review_df)
    pd.testing.assert_frame_equal(std_title_df, expected_title_df)


# Checkpoint 4
def test_split_dataset():
    df, _ = A_Explore_Preprocess_Data.clean_data(student_df)
    df = A_Explore_Preprocess_Data.remove_punctuation(df, ['reviews', 'title'])
    df = A_Explore_Preprocess_Data.word_count_encoder(df, 'reviews', [])
    df = B_Train_Model.set_pos_neg_reviews(df, 3)
    s_train_x, s_val_x, s_train_y, s_val_y = B_Train_Model.split_dataset(
        df, 1, 'sentiment', 'Word Count')

    expected_val_indices = [439, 111, 1047,
                            1176, 70, 352, 289, 374, 930, 167, 912]
    expected_train_indices = set(df.index.values) - set(expected_val_indices)

    assert list(s_val_x.index.values) == expected_val_indices
    assert list(s_val_y.index.values) == expected_val_indices
    assert set(s_train_x.index.values) == expected_train_indices
    assert set(s_train_y.index.values) == expected_train_indices


def preprocess_model_tests(df):
    df, _ = A_Explore_Preprocess_Data.clean_data(df)
    df = A_Explore_Preprocess_Data.remove_punctuation(df, ['reviews', 'title'])
    df = A_Explore_Preprocess_Data.word_count_encoder(df, 'reviews', [])
    df = B_Train_Model.set_pos_neg_reviews(df, 3)
    X_train, X_val, y_train, y_val = B_Train_Model.split_dataset(
        df, 1, 'sentiment', 'Word Count')

    return X_train, X_val, y_train, y_val


# Checkpoint 5
def test_train_logistic_regression():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    params = {'max_iter': 1000,
              'penalty': 'l1', 'tol': 0.01, "solver": "liblinear"}
    student_lr = B_Train_Model.train_logistic_regression(
        X_train, y_train, 'Logistic Regression', params)

    student_pred = student_lr.predict_proba(X_val)
    assert np.allclose(student_pred,
                       np.array([[0.00033509657359, 0.99966490342641],
                                 [0.00006836217688, 0.99993163782312],
                                 [0.06983473692013, 0.93016526307987],
                                 [0.97433884386458, 0.02566115613542],
                                 [0.02602010321553, 0.97397989678447],
                                 [0.00001989392376, 0.99998010607624],
                                 [0.00159098063393, 0.99840901936607],
                                 [0.00536366659173, 0.99463633340827],
                                 [0.00047812840791, 0.99952187159209],
                                 [0.00011844250899, 0.99988155749101],
                                 [0.04846528969827, 0.95153471030173]]))


# Checkpoint 6
def test_train_sgd_classifier():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    params = {'loss': 'log', 'max_iter': 1000, ##
              'penalty': 'l1', 'tol': 0.01, 'alpha': 0.001}
    student_sgd = B_Train_Model.train_sgd_classifer(
        X_train, y_train, 'Stochastic Gradient Descent', params)
    student_pred = student_sgd.predict_proba(X_val)
    assert np.allclose(student_pred,
                       np.array([[0, 1.00000000e+00],
                                 [0, 1.00000000e+00],
                                 [0, 1],
                                 [0.99914037981499, 0.00085962018501],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [0.00038083151739, 0.99961916848261]]))


# Checkpoint 7
def test_train_sgdcv_classifier():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    params = {'loss': ['log'], 'max_iter': [1000], ##
              'penalty': ['l1'], 'tol': [0.01], 'alpha': [0.01]}
    cv_params = {'n_splits': 3, 'n_repeats': 10}
    student_sgdcv = B_Train_Model.train_sgdcv_classifer(
        X_train, y_train, 'Stochastic Gradient Descent with Cross Validation', params, cv_params)
    student_pred = student_sgdcv.predict_proba(X_val)
    assert np.allclose(student_pred,
                       np.array([[0.0007548362424381061, 0.9992451637575619],
                                 [0.00019686647330641183, 0.9998031335266936],
                                 [0.019171329267051118, 0.9808286707329489],
                                 [0.3401483248140079, 0.6598516751859921],
                                 [0.034338006260975606, 0.9656619937390244],
                                 [0.000008949278585634879, 0.9999910507214144],
                                 [0.0000082969090725582, 0.9999917030909274],
                                 [0.041324907723753745, 0.9586750922762463],
                                 [0.014192901114390089, 0.9858070988856099],
                                 [0.004274049342153141, 0.9957259506578469],
                                 [0.022909189548154574, 0.9770908104518454]]))


# Checkpoint 9
def test_inspect_coefficients():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    params_lr = {'max_iter': 1000,
                 'penalty': 'l1', 'tol': 0.01, "solver": "liblinear"}
    student_lr = B_Train_Model.train_logistic_regression(
        X_train, y_train, 'Logistic Regression', params_lr)

    params_sgd = {'loss': 'log', 'max_iter': 1000, ##
                  'penalty': 'l1', 'tol': 0.01, 'alpha': 0.01}
    student_sgd = B_Train_Model.train_sgd_classifer(
        X_train, y_train, 'Stochastic Gradient Descent with Logistic Regression', params_sgd)

    params_sgdcv = {'loss': ['log'], 'max_iter': [1000],  ##
                    'penalty': ['l1'], 'tol': [0.01], 'alpha': [0.01]}
    cv_params = {'n_splits': 3, 'n_repeats': 10}
    student_sgdcv = B_Train_Model.train_sgdcv_classifer(
        X_train, y_train, 'Stochastic Gradient Descent with Cross Validation', params_sgdcv, cv_params)

    trained_models = {'Logistic Regression': student_lr,
                      'Stochastic Gradient Descent with Logistic Regression': student_sgd,
                      'Stochastic Gradient Descent with Cross Validation': student_sgdcv}

    student_coef = B_Train_Model.inspect_coefficients(trained_models)

    exp_lr_coef = np.load('./test_files/lr_coef.npy')
    exp_sgd_coef = np.load('./test_files/sgd_coef.npy')
    exp_sgdcv_coef = np.load('./test_files/sgdcv_coef.npy')

    assert np.allclose(student_coef['Logistic Regression'], exp_lr_coef)
    assert np.allclose(
        student_coef['Stochastic Gradient Descent with Logistic Regression'], exp_sgd_coef)
    assert np.allclose(
        student_coef['Stochastic Gradient Descent with Cross Validation'], exp_sgdcv_coef)


# Checkpoint 10
def test_metrics():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    params = {'max_iter': 1000,
              'penalty': 'l1', 'tol': 0.01, "solver": "liblinear"}
    student_lr = B_Train_Model.train_logistic_regression(
        X_train, y_train, 'Logistic Regression', params)

    student_metrics = C_Test_Model.compute_eval_metrics(
        X_train, y_train, student_lr, ['precision', 'recall', 'accuracy'])

    assert np.allclose(student_metrics['precision'], 0.9847405900305188)
    assert np.allclose(student_metrics['recall'], 1.0)
    assert np.allclose(student_metrics['accuracy'], 0.9856046065259118)


# Checkpoint 11
def test_plot_roc_curve():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)

    params_lr = {'max_iter': 1000,
                 'penalty': 'l1', 'tol': 0.01, "solver": "liblinear"}
    student_lr = B_Train_Model.train_logistic_regression(
        X_train, y_train, 'Logistic Regression', params_lr)

    params_sgd = {'loss': 'log', 'max_iter': 1000, ##
                  'penalty': 'l1', 'tol': 0.01, 'alpha': 0.01}
    student_sgd = B_Train_Model.train_sgd_classifer(
        X_train, y_train, 'Stochastic Gradient Descent', params_sgd)

    _, student_roc_df = C_Test_Model.plot_roc_curve(
        X_train, X_val, y_train, y_val, {'Logistic Regression': student_lr, 'Stochastic Gradient Descent': student_sgd}, ['Logistic Regression', 'Stochastic Gradient Descent'])

    expected_roc_df = pd.read_pickle('./test_files/roc_dict.pkl')
    pd.testing.assert_frame_equal(student_roc_df, expected_roc_df)


# Helper Function
def decode_integer(original_df, decode_df, feature_name):
    """
    Decode integer integer encoded feature

    Input: 
        - original_df: pandas dataframe with feature to decode
        - decode_df: dataframe with feature to decode 
        - feature: feature to decode
    Output: 
        - decode_df: Pandas dataframe with decoded feature
    """
    # original_dataset[[feature_name]]= enc.fit_transform(original_dataset[[feature_name]])
    decode_df[[feature_name]] = enc.inverse_transform(
        st.session_state['X_train'][[feature_name]])
    return decode_df
