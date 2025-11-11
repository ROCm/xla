#include "xla/backends/profiler/gpu/svm_wrapper.h"

#include <svm.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

namespace xla::profiler {

SVMModel::SVMModel() {
  // Initialize problem struct to avoid dangling pointers
  problem.l = 0;
  problem.y = nullptr;
  problem.x = nullptr;
  
  initializeParameters();
}

SVMModel::~SVMModel() {
  cleanup();
}

void SVMModel::initializeParameters() {
  param.svm_type = C_SVC;
  param.kernel_type = LINEAR;
  param.degree = 3;
  param.gamma = 0.0;
  param.coef0 = 0.0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.C = 1.0;
  param.eps = 1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = nullptr;
  param.weight = nullptr;
}

bool SVMModel::train(const std::vector<Point>& points) {
  if (points.empty()) {
    fprintf(stderr, "SVMModel::train: No points\n");
    return false;
  }

  cleanupTrainingData();

  problem.l = points.size();
  problem.y = new double[problem.l];
  problem.x = new svm_node*[problem.l];  // Array of pointers to per-point node arrays

  // FIXED: Flat vector of svm_node structs (contiguous memory)
  x_space.resize(problem.l * 3);
  size_t node_idx = 0;  // Use size_t to match vector

  for (size_t i = 0; i < points.size(); ++i) {
    problem.y[i] = points[i].label;

    // Point to the start of this point's nodes in the flat vector
    problem.x[i] = &x_space[node_idx];

    // Feature 1: x (index 1, 1-based)
    x_space[node_idx].index = 1;
    x_space[node_idx].value = points[i].x;
    ++node_idx;

    // Feature 2: y (index 2, 1-based)
    x_space[node_idx].index = 2;
    x_space[node_idx].value = points[i].y;
    ++node_idx;

    // Terminator for this point's nodes
    x_space[node_idx].index = -1;
    x_space[node_idx].value = 0.0;
    ++node_idx;
  }

  const char* err = svm_check_parameter(&problem, &param);
  if (err) {
    fprintf(stderr, "SVM param error: %s\n", err);
    return false;
  }

  model = svm_train(&problem, &param);
  if (!model) {
    fprintf(stderr, "SVM train failed\n");
    return false;
  }

  // Accuracy (optional, for validation)
  int correct = 0;
  double* pred = new double[problem.l];
  for (int i = 0; i < problem.l; ++i) {
    pred[i] = svm_predict(model, problem.x[i]);
    if (pred[i] == problem.y[i]) ++correct;
  }
  training_accuracy = (100.0 * correct) / problem.l;
  delete[] pred;

  printf("SVM trained: %zu points, acc=%.2f%%\n", points.size(), training_accuracy);
  return true;
}

bool SVMModel::getCoefficients(double coef[2]) const {
  if (!model || model->nr_class != 2 || model->l == 0 || !model->SV || !model->sv_coef) {
    fprintf(stderr, "Invalid model for coef extraction\n");
    coef[0] = coef[1] = 0.0;
    return false;
  }

  coef[0] = coef[1] = 0.0;
  for (int i = 0; i < model->l; ++i) {
    double alpha = model->sv_coef[0][i];
    svm_node* sv = model->SV[i];
    while (sv && sv->index != -1) {
      if (sv->index == 1) coef[0] += alpha * sv->value;
      else if (sv->index == 2) coef[1] += alpha * sv->value;
      ++sv;
    }
  }
  printf("Coef: x=%.15e, y=%.15e\n", coef[0], coef[1]);
  return true;
}

double SVMModel::getIntercept() const {
  return model ? -model->rho[0] : 0.0;
}

double SVMModel::getAlpha() const {
  double coef[2];
  if (!getCoefficients(coef)) {
    return 0.0;
  }
  return -coef[0] / coef[1];
}

double SVMModel::getBeta() const {
  double coef[2];
  if (!getCoefficients(coef)) {
    return 0.0;
  }
  auto intercept = getIntercept();
  return -intercept / coef[1];
}

double SVMModel::getTrainingAccuracy() const {
  return training_accuracy;
}

void SVMModel::cleanup() {
  if (model) {
    svm_free_and_destroy_model(&model);
    model = nullptr;
  }
  cleanupTrainingData();
}

void SVMModel::cleanupTrainingData() {
  delete[] problem.y; problem.y = nullptr;
  delete[] problem.x; problem.x = nullptr;
  x_space.clear();
  problem.l = 0;
}

}  // namespace xla::profiler
