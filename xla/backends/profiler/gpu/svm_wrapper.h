#ifndef XLA_BACKENDS_PROFILER_GPU_SVM_WRAPPER_H_
#define XLA_BACKENDS_PROFILER_GPU_SVM_WRAPPER_H_

#include <vector>
#include <cstdint>

extern "C" {
#include <svm.h>
}

namespace xla::profiler {

struct Point {
  double x, y;
  int label;
};


struct ScaleInfo{
  double y_max = 0.0, y_min = 0.0;
  double x_max = 0.0, x_min = 0.0;
};
struct ProbInfo{
  std::vector<Point> points;
  ScaleInfo scale_info;
};

class SVMModel {
 public:
  SVMModel();
  ~SVMModel();

  bool train(const ProbInfo& prob_info);

  bool getCoefficients(double coef[2]) const;
  double getIntercept() const;
  double getAlpha() const;
  double getBeta() const;
  double getTrainingAccuracy() const;

 private:
  void initializeParameters();
  void cleanup();
  void cleanupTrainingData();

  svm_model* model = nullptr;
  svm_problem problem;
  svm_parameter param;
  std::vector<svm_node> x_space;
  double training_accuracy = 0.0;
  ScaleInfo scale_info;
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_SVM_WRAPPER_H_
