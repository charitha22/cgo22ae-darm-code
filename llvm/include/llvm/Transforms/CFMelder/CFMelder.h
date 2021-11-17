#ifndef LLVM_TRANSFORMS_CFMEGER_CFMERGER_H
#define LLVM_TRANSFORMS_CFMEGER_CFMERGER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class CFMelderPass
    : public PassInfoMixin<CFMelderPass> {

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};


FunctionPass *createCFMelderPass();
}

#endif

