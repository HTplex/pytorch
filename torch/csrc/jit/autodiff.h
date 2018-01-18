#pragma once

#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch { namespace jit {

using value_list = std::vector<Value*>;

std::vector<Value*> gradientForNode(Node* node, ArrayRef<Value*> grad_values);
void differentiate(std::shared_ptr<Graph>& graph);
std::pair<std::shared_ptr<Graph>, value_list> backwardLambdaLift(std::shared_ptr<Graph>& graph);

}}
