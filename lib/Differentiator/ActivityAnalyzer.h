#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <map>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <iterator>
#include <iostream>

using namespace clang;

namespace clad{
class VariedAnalyzer : public clang::RecursiveASTVisitor<VariedAnalyzer>{


    bool m_Varied = false;
    bool m_Marking = false;
    bool m_shouldPushSucc = true;

    std::set<const clang::VarDecl*>& m_VariedDecls;

    struct VarsData {
        std::set<const clang::VarDecl*> m_Data;
        VarsData* m_Prev = nullptr;

        VarsData() = default;
        VarsData(const VarsData& other) = default;
        ~VarsData() = default;
        VarsData(VarsData&& other) noexcept
            : m_Data(std::move(other.m_Data)), m_Prev(other.m_Prev) {}
        VarsData& operator=(const VarsData& other) = delete;
        VarsData& operator=(VarsData&& other) noexcept {
        if (&m_Data == &other.m_Data) {
            m_Data = std::move(other.m_Data);
            m_Prev = other.m_Prev;
        }
        return *this;
        }

        bool operator==(VarsData other) noexcept{
            std::vector<const clang::VarDecl*> diff;
            if(m_Data == other.m_Data)
                return true;
            return false;
        }

        using iterator =
            std::set<const clang::VarDecl*>::iterator;
        int size(){return m_Data.size();}
        iterator begin() { return m_Data.begin(); }
        iterator end() { return m_Data.end(); }
        iterator find(const clang::VarDecl* VD) { return m_Data.find(VD); }
        void insert(const clang::VarDecl* VD){m_Data.insert(VD);}
        void clear() { m_Data.clear(); }
        std::set<const clang::VarDecl*> updateLoopMem(){return m_Data;}
    };

    VarsData m_LoopMem;
    clang::CFGBlock* getCFGBlockByID(unsigned ID);
    // VarData* getExprVarData(const clang::Expr* E, bool addNonConstIdx = false);

    std::set<const clang::VarDecl*> static collectDataFromPredecessors(VarsData* varsData,
                                                    VarsData* limit = nullptr);

    static VarsData* findLowestCommonAncestor(VarsData* varsData1,
                                            VarsData* varsData2);

    void merge(VarsData* targetData, VarsData* mergeData);
    ASTContext& m_Context;
    std::unique_ptr<clang::CFG> m_CFG;
    std::vector<VarsData*> m_BlockData;
    std::vector<short> m_BlockPassCounter;
    unsigned m_CurBlockID{};
    std::set<unsigned> m_CFGQueue;

    void addToVaried(const clang::VarDecl* VD);
    bool isVaried(const clang::VarDecl* VD);
    
    void copyVarToCurBlock(const clang::VarDecl* VD);
    VarsData& getCurBlockVarsData() { return *m_BlockData[m_CurBlockID]; }

public:
    /// Constructor
    VariedAnalyzer(ASTContext& Context, std::set<const clang::VarDecl*>& Decls)
        : m_VariedDecls(Decls), m_Context(Context) {}

    /// Destructor
    ~VariedAnalyzer() = default;

    /// Delete copy/move operators and constructors.
    VariedAnalyzer(const VariedAnalyzer&) = delete;
    VariedAnalyzer& operator=(const VariedAnalyzer&) = delete;
    VariedAnalyzer(const VariedAnalyzer&&) = delete;
    VariedAnalyzer& operator=(const VariedAnalyzer&&) = delete;

    /// Visitors
    void Analyze(const clang::FunctionDecl* FD);

    void VisitCFGBlock(const clang::CFGBlock& block);

    bool VisitBinaryOperator(clang::BinaryOperator* BinOp);
    bool VisitCallExpr(clang::CallExpr* CE);
    bool VisitConditionalOperator(clang::ConditionalOperator* CO);
    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    bool VisitDeclStmt(clang::DeclStmt* DS);
    bool VisitUnaryOperator(clang::UnaryOperator* UnOp);
};
}