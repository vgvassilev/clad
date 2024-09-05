#include "ActivityAnalyzer.h"

using namespace clang;

namespace clad{

void VariedAnalyzer::Analyze(const FunctionDecl* FD) {  
    // Build the CFG (control-flow graph) of FD.
    clang::CFG::BuildOptions Options;
    m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), &m_Context, Options);


    m_BlockData.resize(m_CFG->size());
    m_BlockPassCounter.resize(m_CFG->size(), 0);

    // Set current block ID to the ID of entry the block.
    auto* entry = &m_CFG->getEntry();
    m_CurBlockID = entry->getBlockID();
    m_BlockData[m_CurBlockID] = new VarsData();
    for(const auto& i: m_VariedDecls){
      m_BlockData[m_CurBlockID]->insert(i);
    }
    // Add the entry block to the queue.
    m_CFGQueue.insert(m_CurBlockID);

    // Visit CFG blocks in the queue until it's empty.
    while (!m_CFGQueue.empty()) {
        auto IDIter = std::prev(m_CFGQueue.end());
        m_CurBlockID = *IDIter;
        m_CFGQueue.erase(IDIter);
        CFGBlock& nextBlock = *getCFGBlockByID(m_CurBlockID);
        VisitCFGBlock(nextBlock);
    }
}

CFGBlock* VariedAnalyzer::getCFGBlockByID(unsigned ID) {
  return *(m_CFG->begin() + ID);
}

void VariedAnalyzer::VisitCFGBlock(const CFGBlock& block) {
  // Visit all the statements inside the block.
  for (const clang::CFGElement& Element : block) {
      if (Element.getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = Element.castAs<clang::CFGStmt>().getStmt();
      TraverseStmt(const_cast<clang::Stmt*>(S));
      }
  }

  for(const auto succ: block.succs()){
    if (!succ)
      continue;
    auto& succData = m_BlockData[succ->getBlockID()];

    if(!succData){
      succData = new VarsData(*m_BlockData[block.getBlockID()]);
      succData->m_Prev = m_BlockData[block.getBlockID()];
    }


    if(succ->getBlockID() > block.getBlockID()){
      if(m_LoopMem == *m_BlockData[block.getBlockID()])
        m_shouldPushSucc = false;
      
      // has to be changed
      for(const auto& i : *m_BlockData[block.getBlockID()])
        m_LoopMem.insert(i);
    }

    if(m_shouldPushSucc){
      m_CFGQueue.insert(succ->getBlockID());
    }
    m_shouldPushSucc = true;

    merge(succData, m_BlockData[block.getBlockID()]);
  }

  // has to be changed
  for(const auto& i: *m_BlockData[block.getBlockID()])
    m_VariedDecls.insert(i);
}

bool VariedAnalyzer::isVaried(const VarDecl* VD){
    auto& curBranch = getCurBlockVarsData();
    return curBranch.find(VD) != curBranch.end();
}

void VariedAnalyzer::merge(VarsData* targetData, VarsData* mergeData) {
  for(const auto& i: *mergeData){
    targetData->insert(i);
  }
  for(const auto& i: *targetData){
    mergeData->insert(i);
  }
}

void VariedAnalyzer::copyVarToCurBlock(const clang::VarDecl* VD) {
    auto& curBranch = getCurBlockVarsData();
    curBranch.insert(VD);
}

bool VariedAnalyzer::VisitBinaryOperator(BinaryOperator* BinOp) {
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();

  if(BinOp->isAssignmentOp()){
    m_Varied = false;
    TraverseStmt(R);
    m_Marking = m_Varied;
    TraverseStmt(L);
    m_Marking = false;
  }else{
    TraverseStmt(L);
    TraverseStmt(R);
  }
  return true;
}

bool VariedAnalyzer::VisitConditionalOperator(clang::ConditionalOperator* CO) {
  return true;
}

bool VariedAnalyzer::VisitDeclStmt(DeclStmt* DS) {
  for (auto* D : DS->decls()){
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      if(Expr* init = VD->getInit()){
        m_Varied = false;
        TraverseStmt(init);
        m_Marking = true;
        auto& curBranch = getCurBlockVarsData();
        if(curBranch.find(VD) == curBranch.end() && m_Varied){
          copyVarToCurBlock(VD);
        }
        m_Marking = false;
      }
    }
  }
  return true;
}


bool VariedAnalyzer::VisitUnaryOperator(UnaryOperator* UnOp){
  Expr* E = UnOp->getSubExpr();
  TraverseStmt(E);
  return true;
}

bool VariedAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
    if(isVaried(dyn_cast<VarDecl>(DRE->getDecl()))){
        m_Varied = true;
    }

    if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        auto& curBranch = getCurBlockVarsData();
        if (curBranch.find(VD) == curBranch.end() && m_Varied && m_Marking)
            copyVarToCurBlock(VD);
    }
    return true;
}
}
