#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"
#include <cmath>

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;
    int seed_;
    double prediction;
    double Error;
    double Update;
    
public:
    PerceptronFeatureHashing(int log_num_buckets, double learning_rate)
        : log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0x9748cd)
    {
        // set all weights to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email)
    {
        std::vector<double> feature_vector_;
        feature_vector_.resize(1 << log_num_buckets_, 0.0);
        for (int i=0; i<=(email.num_words()-ngram_k); i++)
        {
            feature_vector_[get_bucket(email.get_ngram(i,ngram_k))] += 1;
        }
        prediction = predict_(email);

        if(email.is_spam())
        {  
            Error = 1 - prediction;
            Update =  Error * learning_rate_; // possibility to add the activation function here
            bias_+=Update;
            for (int i=0; i<feature_vector_.size(); i++)
            {
                weights_[i] = weights_[i] + (Update*feature_vector_[i]);
            }
         }
        
        else
        {  
            Error = (-1) - prediction;
            Update =  Error * learning_rate_; // possibility to add the activation function here
            bias_+=Update;
            for (int i=0; i<feature_vector_.size(); i++)
            {
                weights_[i] = weights_[i] + (Update*feature_vector_[i]);
            }
         }
    }
 
    double predict_(const Email& email) const
    {
       std::vector<double> feature_vector2_;
       double sum = 0.0;
       feature_vector2_.resize(1 << log_num_buckets_, 0.0);
        
        for (int i=0; i<=(email.num_words()-ngram_k); i++)
        {
            feature_vector2_[get_bucket(email.get_ngram(i,ngram_k))] += 1;
        }
        for(int i=0; i<feature_vector2_.size(); i++)
        {
             sum = sum + (feature_vector2_[i])*(weights_[i]);
        } 
        sum = sum + bias_;
        return 2.0/(1.0+(exp(-2*sum)))-1.0;
    }

    void print_weights() const
    {
        std::cout << "bias " << bias_ << std::endl;
        for (size_t i = 0; i < weights_.size(); ++i)
        {
            std::cout << "w" <<i << " " << weights_[i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        // TODO limit the range of the hash values here
        hash = hash % (1 << log_num_buckets_);
        return hash;
    }
};

} // namespace bdap
