# Approach:

### 1.0 - EDA
First, I loaded the data and took a look, I realized that we have two goal columns, `final_exam_score` & `final_grade`.

Since we want our problem to be **Classification**, what we're going to predict is going to be the `final_grade` column because it has multiple classes; A, B, C, D and F.

### 2.0 - Feature Engineering
After that conclusion I decided to engineer our `final_grade` column to be more precise, so each grade now is separated into two grades (except the F grade, it remains the same), the regular grade and the grade with `+`, the way I separated them is by using the `final_exam_score` and applying it to the `get_grade(exam_score)` function which checks if the score meets certain conditions to get the final grade (with the + or without).

We had 5 classes but after what we've done, now we have 9 (A+, A, B+, B, C+, C, D+, D, F).

### 3.0 - Pre-Processing
Now we move to the pre-processing part.

### 3.1 - Filling missing data
The data has some missing values in the `parental_education` column, and we do not have a dominant Mode to fill the data with, and the data is so little that I can't drop any samples and waste them; I created a new class named `Unknown` and filled the missing values in that column with it.

It contained two sections (Embedding & Standardization).
First of all, I created 3 lists:

- First list for the categorical (string) columns (e.g. `parental_education`).

- Second list of the binary columns (e.g. `gender`) that are either 0 or 1.

- Third list of the numerical columns (e.g. `sleep_hours`).

### 3.2 - Embedding
- Binary columns: I embedded the binary columns (e.g. `Yes` & `No` to `0` & `1`) using `LabelEncoder()`.

- Categorical columns: I embedded them using the built-in pandas function `get_dummies`.

### 3.3 - Standardization
- Numerical columns: They don't need embedding (clearly because they're already numerical), but they DO NEED some work — I applied `StandardScaler()` to scale them all around 0 so larger values do not dominate the smaller ones.

### 4.0 - Models Training
After everything we've done, now we're ready to get our hands on the model. I trained the model on the new embedded & scaled split, but it wasn't a regular training — I used `GridSearchCV` with some parameters for each of my two models, `Logistic Regression` and `Random Forest Classifier`.

### 4.1 - GridSearchCV & StratifiedKFold
I used `StratifiedKFold()` to split my training data into 5 parts.

### 5.0 - Poor scoring on evaluation (imbalanced classes)
Because of my feature engineering on the goal column, it divided into smaller parts (5 → 9), and my data was already too small, so there wasn't a lot of data and the model was overfitting. So I just reversed my feature engineering on the goal column to what it was before my processing (9 → 5), but even that wasn't enough because the dataset is still too small, being only 1000 samples.

- Logistic Regression F1 score: ~53%
- Random Forest Classifier F1 score: ~49%

### 6.0 - Synthetic Data
I created many more samples myself using the `K-NN` algorithm via `SMOTE`, resulting in **3150** samples in training instead of only **800**.
Now we cover all the rare classes like the `F` class, which we only had **12** samples of, but after our resampling now we have **150** of it, so we can give our models some space to learn the patterns, and now we've fixed the class imbalancing.

### 7.0 - Re-train models & Re-evaluate them
After all we've been through, now we retrain the models and test them on our new, much larger data, and it wasn't a surprise that the scores went up significantly, but our `LogisticRegression()` model still isn't doing very well, even though it improved.
- Logistic Regression F1 score: ~62%
- Random Forest Classifier F1 score: ~92%

Of course the `RandomForestClassifier()` had a big jump because its algorithm is very "thirsty" for data and it performs very well on larger datasets.

### 8.0 - Save the model
We need to save the model so we can use it in Gradio for the next phase; I used `pickle` to save the model.
> Note: I saved my best model, which is the `RandomForestClassifier()`.

### 9.0 - Gradio User Interface (UI)
I used `Gradio` for the UI, and my model is `RandomForestClassifier()` since it outperformed `LogisticRegression()`.
<img width="1686" height="705" alt="image" src="https://github.com/user-attachments/assets/b35d99ce-5c11-4ee7-9f5d-17d7fbf92c2f" />
