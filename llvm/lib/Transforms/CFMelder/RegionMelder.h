#ifndef LLVM_LIB_TRANSFORMS_REGION_MELDER_H
#define LLVM_LIB_TRANSFORMS_REGION_MELDER_H

#include "RegionAnalyzer.h"
#include "InstructionMatch.h"
#include "SmithWaterman.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/TableGen/Record.h"

namespace llvm {


// simple scoring function for intruction alignment
struct InstrMeldingProfitabilityModel : public ScoringFunction<Value *> {
public:
  int operator()(Value *V1, Value *V2) override {


    if (!InstructionMatch::match(V1, V2))
      return 0;

    int SavedCycles = 0;
    if (isa<Instruction>(V1)) {
      Instruction *I1 = dyn_cast<Instruction>(V1);
      SavedCycles = InstructionMatch::getInstructionCost(I1);
    }
    else if (isa<BasicBlock>(V1)) {
      SavedCycles = 3;
    }

    return SavedCycles;
  }
  int gap(int K) override { return 2; }
};

class RegionMelder {
private:
  RegionAnalyzer &MA;

  // entry and exits of currently merging regions
  // if single BB, exit is null
  BasicBlock *EntryBlockL{nullptr}, *EntryBlockR{nullptr};
  BasicBlock *ExitBlockL{nullptr}, *ExitBlockR{nullptr};
  DenseMap<BasicBlock *, BasicBlock *> CurrMapping;

  AlignedSeq<Value *> RegionInstrAlignement;
  using ValueToValueMapTy = ValueMap<const Value *, WeakTrackingVH>;

  // value maps
  ValueToValueMapTy MergedValuesToLeftValues;
  ValueToValueMapTy MergedValuesToRightValues;
  ValueToValueMapTy OrigToMergedValues;

  // label map
  DenseMap<BasicBlock *, BasicBlock *> LeftBbToMergedBb;
  DenseMap<BasicBlock *, BasicBlock *> RightBbToMergedBb;

  // merged instructions
  SmallVector<Instruction *, 0> MergedInstructions;
  SmallVector<BasicBlock *, 0> MergedBBs;

  // un-predication
  struct InstrRange {
  private:
    Instruction *StartI, *EndI;
    bool Direction;

  public:
    InstrRange(Instruction *Start, Instruction *End, bool Dir)
        : StartI(Start), EndI(End), Direction(Dir) {}
    Instruction *getStart() const { return StartI; }
    Instruction *getEnd() const { return EndI; }
    bool splitToTrue() const { return Direction; }
    bool splitToFalse() const { return !Direction; }
    friend raw_ostream &operator<<(raw_ostream &OS, const InstrRange &Range) {
      OS << "start : ";
      Range.getStart()->print(OS);
      OS << ", end : ";
      Range.getEnd()->print(OS);
      OS << ", direction : " << (Range.splitToTrue() ? "T" : "F");
      return OS;
    }

    bool canExtendUsing(const InstrRange &Other) {
      return Other.getStart()->getPrevNode() == EndI &&
             Other.splitToTrue() == splitToTrue();
    }
    InstrRange extendUsing(const InstrRange &Other) {
      assert(canExtendUsing(Other) &&
             "this range can not be extended with provided range !");
      return InstrRange(StartI, Other.getEnd(), splitToTrue());
    }
  };

  SmallVector<InstrRange, 16> SplitRanges;

  AlignedSeq<Value *> getAlignmentOfBlocks(BasicBlock *LeftBb,
                                           BasicBlock *RightBb);
  void computeRegionSeqAlignment(DenseMap<BasicBlock *, BasicBlock *> BbMap);
  void linearizeBb(BasicBlock *BB, SmallVectorImpl<Value *> &LinearizedVals);
  void cloneInstructions();
  void fixOperends();
  void runPostMergeCleanup();
  void setOperends(Instruction *LeftI, Instruction *RightI,
                   Instruction *MergedI);
  void setOperendsForBr(BranchInst *LeftBr, BranchInst *RightBr,
                        BranchInst *MergedBr);
  Instruction *cloneInstruction(Instruction *OrigI, IRBuilder<> &Builder);
  void fixPhiNode(PHINode *Orig);
  void runPostOptimizations();
  void setOprendsForNonMatchingStore(StoreInst *SI, bool IsLeft);
  void runPreMergePasses(bool RegionAlreadySimplified);
  void mergeOutsideDefsAtEntry();

  // makes the region SESE, after simplification exit block of the region
  // is connected to rest of the CFG with only one edge
  BasicBlock *simplifyRegion(BasicBlock *Exit, BasicBlock *Entry);
  
  bool isExitBlockSafeToMerge(BasicBlock *Exit, BasicBlock *Entry);
  void updateMapping(BasicBlock *NewBb, BasicBlock *OldBb, bool IsLeft);

  void runUnpredicationPass();
  void updateSplitRangeMap(bool Direction, Instruction *I);

  bool isInsideMeldedRegion(BasicBlock *BB, BasicBlock* Entry, BasicBlock* Exit);

  // finds the region whose entry block post dominates the path entry block
  Region* getRegionToReplicate(BasicBlock * MatchedBlock, BasicBlock* PathEntry);

public:
  RegionMelder(RegionAnalyzer &MA) : MA(MA) {}
  // meld I'th region pair
  void merge(unsigned Index);

};

} // namespace llvm


#endif