#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int log_num_buckets_;

    int total_num_of_spam = 0;
    int total_num_of_ham = 0;
    int total_num_of_mails = 0;
    int total_words_in_ham_ = 0;
    int total_words_in_spam_ = 0;
    int seed_;
    std::vector<int> number_of_times_word_in_ham_;
    std::vector<int> number_of_times_word_in_spam_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
        : log_num_buckets_(log_num_buckets)
        , seed_(0x249cd)
    {
    //counter_.resize(1 << log_num_buckets_, 1);
    this->threshold = threshold;

    //how many times that word appeared in ham
    number_of_times_word_in_ham_.resize(1 << log_num_buckets_, 1);

    //how many times that word appeared in spam
    number_of_times_word_in_spam_.resize(1 << log_num_buckets_, 1);
    }

    void update_(const Email& email)
    {
        total_num_of_mails+=1;

        if(email.is_spam())
        { 
            total_num_of_spam+=1;

            //converting the mail to hash and adding "1" to counter
            for (int i=0; i<(email.num_words()-ngram_k); i++)
            {
                number_of_times_word_in_spam_[get_bucket(email.get_ngram(i,ngram_k))]+=1;
                total_words_in_spam_ +=1;
            }
        }
        else
        { 
            total_num_of_ham+=1;

            //converting the mail to hash and adding "1" to counter
            for (int i=0; i<=(email.num_words()-ngram_k); i++)
            {
                number_of_times_word_in_ham_[get_bucket(email.get_ngram(i,ngram_k))]+=1;
                total_words_in_ham_ +=1;
            }
        }

    }

    double predict_(const Email& email) const
    {
        int total_num_of_spam_correctly_predicted_as_spam = 0; //True(correclty) negative(ham)
        int total_num_of_ham_correctly_predicted_as_ham = 0; //True(correclty) positive(ham)
        int total_num_of_spam_false_predicted_as_ham = 0; // False(False) positive(ham)
        int total_num_of_ham_false_predicted_as_spam = 0; //False(False) negative(spam)

        std::vector<int> feature_vector2_;
        feature_vector2_.resize(1 << log_num_buckets_, 0);

        //vector of probability for each word of the mail to be in ham
        std::vector<double> pr_word_be_ham_;
        pr_word_be_ham_.resize(1 << log_num_buckets_);

        //vector of probability for each word of the mail to be in spam
        std::vector<double> pr_word_be_spam_;
        pr_word_be_spam_.resize(1 << log_num_buckets_);

        //probabilities of mail be ham or spam
        double pr_mail_be_ham_;
        double pr_mail_be_spam_;

        //product of each word's individual probability 
        // to be ham or spam, for all the words of a single email
        double product_h = 0;
        double product_s = 0;

        double ratio_ham;
        double ratio_spam;

        //initialization of th word vector
        for (int i=0; i<=(email.num_words()-ngram_k); i++)
        {
            feature_vector2_[get_bucket(email.get_ngram(i,ngram_k))] += 1;
        }
       
        //probabilities of each word to be in ham or spam
        for(int i=0; i<feature_vector2_.size(); i++)
        {
            //pr_word_be_ham_[i] = (number_of_times_word_in_ham_[i])/(total_words_in_ham_);
            pr_word_be_ham_[i] = log(number_of_times_word_in_ham_[i]) - log(total_words_in_ham_);
            //pr_word_be_spam_[i] = (number_of_times_word_in_spam_[i])/(total_words_in_spam_);
            pr_word_be_spam_[i] = log(number_of_times_word_in_spam_[i]) - log(total_words_in_spam_);

            product_h += feature_vector2_[i] * pr_word_be_ham_[i];
            product_s += feature_vector2_[i] * pr_word_be_spam_[i];
        }

        ratio_ham = log(total_num_of_ham) - log(total_num_of_mails+((1<<log_num_buckets_)));
        ratio_spam = log(total_num_of_spam) - log(total_num_of_mails+((1<<log_num_buckets_)));

        pr_mail_be_ham_ = ratio_ham + product_h;
        pr_mail_be_spam_ = ratio_spam + product_s;

        if(pr_mail_be_ham_ >= pr_mail_be_spam_) 
        {
            //mail predicted as ham
            total_num_of_ham_correctly_predicted_as_ham+=1;
        }
        else if (pr_mail_be_ham_ < pr_mail_be_spam_)
        {
            //mail predicted as spam
            total_num_of_spam_correctly_predicted_as_spam+=1;
        }

        //softmax activation conversion
        double Z = (pr_mail_be_ham_)+(pr_mail_be_spam_);
        double calculate_h = (pr_mail_be_ham_);
        double calculate_s = (pr_mail_be_spam_);

        return 1-calculate_s/Z;
    }

    /*void print_weights() const
    {
        size_t n = (1 << log_num_buckets_);
        for (size_t i = 0; i < n; ++i)
        {
            std::cout << "w" <<i << " " << feature_vector_[i] << ", " << feature_vector_[n + i] << std::endl;
        }
    }*/

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        hash = hash % (1 << log_num_buckets_);
        return hash;
    }
};

} // namespace bdap
