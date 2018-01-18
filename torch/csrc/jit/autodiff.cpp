#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit {

using value_list = std::vector<Value*>;

std::vector<Value*> gradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  const auto build_sym_grad = [node](const std::vector<SymbolicVariable>& grads) -> std::vector<SymbolicVariable> {
    auto inputs = node->inputs();
    switch(node->kind()) {
      case kadd:
        return {grads[0], grads[0]};
      case ksub:
        return {grads[0], -grads[0]};
      case kmul:
        return {grads[0] * inputs[1], grads[0] * inputs[0]};
    }
    throw std::runtime_error(std::string("don't support differentiation of `") +
                            node->kind().toString() + "`");
  };
  auto sym_grads = build_sym_grad(fmap<SymbolicVariable>(grad_values));
  return fmap(sym_grads, [](const SymbolicVariable &v) { return v.value(); });
}

void differentiate(std::shared_ptr<Graph>& graph) {
  JIT_ASSERT(graph->stage() == 0);
  graph->advanceStage();

  std::unordered_map<Value*, Value*> grad_map; // x -> dx mapping
  const auto get_grad = [&](Value* v) { return grad_map[v]; };
  for (auto output : graph->outputs())
    grad_map[output] = graph->addInput()->setType(output->typeOption());

  for (auto it = graph->rbegin(), end = graph->rend(); it != end; ++it) {
    Node *node = *it;
    auto inputs = node->inputs();
    value_list grad_inputs = gradientForNode(node, fmap(node->outputs(), get_grad));
    JIT_ASSERT(grad_inputs.size() == node->inputs().size());
    for (std::size_t i = 0, num_inputs = grad_inputs.size(); i < num_inputs; ++i) {
      if (Value * prev_grad = grad_map[inputs[i]]) {
        Node *new_grad_node = graph->create(kadd, {prev_grad, grad_inputs[i]})
                                   ->t_(kalpha, at::Scalar(1).toTensor());
        new_grad_node->insertAfter(grad_inputs[i]->node());
        Value *new_grad = new_grad_node->output();
        new_grad->setType(prev_grad->typeOption());
        grad_map[inputs[i]] = new_grad;
      } else {
        grad_map[inputs[i]] = grad_inputs[i];
      }
    }
  }

  for (auto input : graph->inputs()) {
    if (input->stage() > 0) break;
    graph->registerOutput(grad_map.at(input));
  }
}

// TODO: return metadata that allows to associate inputs with values form graph
std::pair<std::shared_ptr<Graph>, value_list> backwardLambdaLift(std::shared_ptr<Graph>& graph) {
  JIT_ASSERT(graph->stage() == 1);

  std::unordered_set<Value*> backward_inputs_set;
  for (Node *node : *graph) {
    if (node->stage() == 0) continue;
    for (Value *input : node->inputs()) {
      // XXX: this assumes that the single Param node in the graph (that holds inputs) has stage 0
      if (input->node()->stage() == 0)
        backward_inputs_set.insert(input);
    }
  }
  value_list backward_inputs(backward_inputs_set.begin(), backward_inputs_set.end());
  // We want them nicely sorted with inputs being first, and then following
  // the topological ordering (at least approximately, we could do a better job here).
  std::sort(backward_inputs.begin(), backward_inputs.end(),
            [](Value *a, Value*b) { return a->unique() < b->unique(); });

  auto dgraph = std::make_shared<Graph>();
  std::unordered_map<Value*, Value*> val_map; // values in graph -> values in dgraph
  const auto lookup_val = [&](Value *v) { return val_map.at(v); };
  for (auto input : backward_inputs)
    val_map[input] = dgraph->addInput()->setType(input->typeOption());
  for (Node *node : *graph) {
    if (node->stage() == 0) continue;
    Node *clone = dgraph->createClone(node, lookup_val);
    for (std::size_t i = 0, num_outputs = clone->outputs().size(); i < num_outputs; ++i)
      val_map[node->outputs()[i]] = clone->outputs()[i];
    dgraph->appendNode(clone);
  }
  for (auto output : graph->outputs()) {
    if (output->stage() == 0) continue;
    dgraph->registerOutput(val_map.at(output));
  }
  return std::make_pair(dgraph, backward_inputs);
}

}}
