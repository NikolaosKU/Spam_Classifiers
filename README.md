# Spam_Classifiers

The project was completely constructed using Unix-Based environment, solely using Mac Os compatible iTerm2 terminal. The code was build in C++ 17 exclusively in VS Code. Compiling was achieved with Cmake for portability.

This is a project for building and train methods/algorithms making them capable of detecting Spam and Ham on unread incoming emails (in batch).

The Dataset consists of ~250K emails. In the 'data' folder there is an instance of the total dataset.

Each email comes with certain characteristcs:

header: contains metadata such if the mail is spam (label=1) or ham (label=0)
body: content of email

I have built two clasifiers using the following algorithms:

- Naive Bayes
- Perceptron

Each classifier receives a batch of emails (train data) (modifiable), trains that batch according to the classifiers' inherent characteristics and then test the prediction to the next batch (step) of emails (test data). 'Step' is modifiable and can be as small as 1 (my program is trained with a single email and tests the prediction upon the next email) and could be as big as 125000 (my program gets trained with the first 125K emails and then predicts the rest). Both of those step choices are extremes.

A matric.hpp file calcualates the accuracy of each method. Additional metrics can be implemented such as ROC, AUC, F-Score.

To handle the infinite potential features of an email, Naive Bayes and Perceptron were boosted with the techniques:

- feature hashing
- count min

In the end, there are four classifier methods:

- Naive_Bayes_Count_Min.hpp
- Naive_Bayes_Feature_Hashing.hpp
- Perceptron_Count_Min.hpp
- Perceptron_Feature_Hashing.hpp

Performance was tested for all of them so to decide which is more accurate.

! Disclaimer: The code is not my property. Information about the authors are clearly stated in each file !
