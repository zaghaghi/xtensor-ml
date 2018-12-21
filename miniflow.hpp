#ifndef CPP_MINIFLOW_HPP_
#define CPP_MINIFLOW_HPP_
#include <vector>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

using element_type = float;
using tensor_type = xt::xarray<element_type>;

class node : std::enable_shared_from_this<node> {
 protected:
    std::string name;
    tensor_type value;
    std::vector<std::shared_ptr<node>> output_nodes;
    std::vector<std::shared_ptr<node>> input_nodes;
    std::unordered_map<std::string, tensor_type> gradients;

 public:
    explicit node(const std::string &name,
        const std::vector<std::shared_ptr<node>> &input_nodes = {})
        : name(name), input_nodes(input_nodes) {
        for (auto &node : input_nodes) {
            node->output_nodes.push_back(shared_from_this());
        }
    }

    std::string get_name() const {
        return name;
    }
    
    const tensor_type& get_value() {
        return value;
    }
    
    void update_value(element_type learning_rate = 1e-2) {
        value -= learning_rate * gradients[name];
    }

    tensor_type& get_gradient(const std::string& node_name) {
        return gradients[node_name];
    }
    
    tensor_type& get_self_gradient() {
        return gradients[name];
    }
    
    const std::vector<std::shared_ptr<node>>& get_output_nodes() const {
        return output_nodes;
    }
    
    std::vector<std::shared_ptr<node>>& get_output_nodes() {
        return output_nodes;
    }

    std::shared_ptr<node> get_output_node(int idx) {
        return output_nodes[idx];
    }
    
    const std::vector<std::shared_ptr<node>>& get_input_nodes() const {
        return input_nodes;
    }
    
    std::vector<std::shared_ptr<node>>& get_input_nodes() {
        return input_nodes;
    }

    std::shared_ptr<node> get_input_node(int idx) {
        return input_nodes[idx];
    }
    
    void init_input_gradients() {
        for (auto& node : input_nodes) {
            gradients[node->name] = xt::zeros_like(node->value);
        }
    }

    virtual void forward() = 0;
    virtual void backward() = 0;

};

class input: public node {
    std::vector<int> shape;
 public:
    input(const std::string &name, const std::vector<int>& shape):
        node(name), shape(shape) { }

    void forward() override { }

    void backward() override {
        get_self_gradient() = xt::zeros<element_type>(shape);
        for (auto& node : get_output_nodes()) {
            get_self_gradient() += node->get_gradient(get_name());
        }
    }
};

class linear: public node {
 public:
    linear(const std::string &name, const std::shared_ptr<node>& X,
        const std::shared_ptr<node>& W, const std::shared_ptr<node>& b):
        node(name, {X, W, b}) {}

    void forward() override {
        auto& X_value = input_nodes[0]->get_value();
        auto& W_value = input_nodes[1]->get_value();
        auto& b_value = input_nodes[2]->get_value();
        value = xt::linalg::dot(X_value, W_value) + b_value;
    }

    void backward() override {
        init_input_gradients();
        auto &X = input_nodes[0];
        auto &W = input_nodes[1];
        auto &b = input_nodes[2];
        for (auto& node : output_nodes) {
            auto grad_cost = node->get_gradient(name);
            gradients[X->get_name()] += xt::linalg::dot(grad_cost, xt::transpose(W->get_value()));
            gradients[W->get_name()] += xt::linalg::dot(xt::transpose(X->get_value()), grad_cost);
            gradients[b->get_name()] += xt::sum(grad_cost, {0});
        }
    }
};


class sigmoid: public node {
 public:
    sigmoid(const std::string &name, const std::shared_ptr<node> &input):
        node(name, {input}) {}

    void forward() override {
        auto val = input_nodes[0]->get_value();
        value = 1 / (1 + xt::exp(-val));
    }

    void backward() override {
        init_input_gradients();
        for (auto &node : output_nodes) {
            auto& grad_cost = node->get_gradient(name);
            gradients[input_nodes[0]->get_name()] += value * (1 - value) * grad_cost;
        }
    }
};


class mse: public node {
    element_type m;
    tensor_type diff;
 public:
    mse(const std::string& name, const std::shared_ptr<node>& y,
        const std::shared_ptr<node>& a):
        node(name, {y, a}) { }

    void forward() override {
        auto y = xt::reshape_view(input_nodes[0]->get_value(), {-1, 1});
        auto a = xt::reshape_view(input_nodes[1]->get_value(), {-1, 1});
        m = input_nodes[0]->get_value().shape()[0];
        diff = y - a;
        value = xt::mean(xt::pow(diff, 2));
    }

    void backward() override {
        gradients[input_nodes[0]->get_name()] = (2 / m) * diff;
        gradients[input_nodes[1]->get_name()] = (-2 / m) * diff;
    }
};

std::vector<std::shared_ptr<node>>
topological_sort(const std::vector<std::shared_ptr<node>> & feed_dict) {
    std::unordered_map<std::string,
        std::pair<std::unordered_set<std::string>, std::unordered_set<std::string>>> graph;

    std::stack<std::shared_ptr<node>> nodes;
    for (auto& node : feed_dict) {
        nodes.push(node);
    }
    while (!nodes.empty()) {
        auto node = nodes.top();
        nodes.pop();
        if (graph.count(node->get_name()) == 0) {
            graph[node->get_name()] = std::make_pair(
                std::unordered_set<std::string>(),
                std::unordered_set<std::string>());
        }
        for (auto& output_node : node->get_output_nodes()) {
            if (graph.count(output_node->get_name()) == 0) {
                graph[output_node->get_name()] = std::make_pair(
                    std::unordered_set<std::string>(),
                    std::unordered_set<std::string>());
            }
            graph[node->get_name()].second.insert(output_node->get_name());
            graph[output_node->get_name()].first.insert(node->get_name());
            nodes.push(output_node);
        }
    }

    std::vector<std::shared_ptr<node>> sorted_nodes;
    std::unordered_set<std::shared_ptr<node>> nodes_set(
        feed_dict.begin(), feed_dict.end());

    while (!nodes_set.empty()) {
        auto& cur_node = *nodes_set.begin();
        nodes_set.erase(nodes_set.begin());

        // TODO(zaghaghi): if name is in feed_values set node value

        sorted_nodes.push_back(cur_node);
        for (auto& out_node : cur_node->get_output_nodes()) {
            std::unordered_set<std::string>&
                node_out = graph[cur_node->get_name()].second;
            std::unordered_set<std::string>&
                out_node_in = graph[out_node->get_name()].first;
            node_out.erase(out_node->get_name());
            out_node_in.erase(cur_node->get_name());
            if (out_node_in.empty()) {
                nodes_set.insert(out_node);
            }
        }
    }

    return sorted_nodes;
}

void forward_backward(std::vector<std::shared_ptr<node>> sorted_graph) {
    for (auto& node : sorted_graph) {
        node->forward();
    }

    for (auto it=sorted_graph.rbegin(); it != sorted_graph.rend(); ++it) {
        (*it)->backward();
    }
}

void sgd_update(std::vector<std::shared_ptr<node>> trainables,
    element_type learning_rate = 1e-2) {
    for (auto& node : trainables) {
        node->update_value(learning_rate);
    }
}
#endif  // CPP_MINIFLOW_HPP_
