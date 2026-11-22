#include "row_neighborhood_kernel.cpp"

extern "C" PS_EXPORT char* block_neighborhood(int n,const uint8_t*info,const uint8_t*supports,
                     const uint8_t*removed,int cutoff,uint64_t limit) {
    std::string result;
    try {
        if(n<4||n>ps::capacity||(n&(n-1))||!info||!supports||!removed||cutoff<1||cutoff>n||!limit)
            throw std::invalid_argument("Invalid block-neighborhood input");
        std::vector<ps::Bits> checks,removed_checks;int dimension=0;
        for(int j=0;j<n;++j){
            if(info[j]>1||removed[j]>1||(info[j]&&removed[j]))throw std::invalid_argument("Invalid masks");
            dimension+=info[j];ps::Bits row=ps::unit(j);
            for(int i=0;i<n;++i){
                int b=supports[j*n+i];
                if(b>1||(b&&(info[j]||!info[i]||i>=j)))throw std::invalid_argument("Noncausal support");
                if(b)row.flip(i);
            }
            if(removed[j])removed_checks.push_back(row);else if(!info[j])checks.push_back(row);
        }
#ifdef PS_CAUSAL_INFORMATION_LEADERS
        if(removed_checks.size()>8)throw std::invalid_argument("Expected zero to eight rows");
#else
        if(removed_checks.empty()||removed_checks.size()>8)throw std::invalid_argument("Expected one to eight rows");
#endif
        auto begin=ps::now();auto prep=ps::prepare(n,checks,ps::Bits(),ps::Bits());
        if(prep.transmitted!=dimension+int(removed_checks.size())||prep.kernel)throw std::runtime_error("Supercode dimension mismatch");
        ps::Work work(limit);ps::Solver solver(n,prep.checks,cutoff,0,work);
        std::ostringstream words;uint64_t count=0,parent_target=0,parent_low=0;
        // A word led by a released frozen position violates that restored
        // causal check for every support: all of its historical bits are zero.
#ifdef PS_CAUSAL_INFORMATION_LEADERS
        for(int i=0;i<n;++i)if(info[i])solver.generate(0,ps::Costs(n,1),i,[&](ps::Bits u,ps::Bits x){
#else
        for(int i=0;i<n;++i)if(!prep.checks.count(i))solver.generate(0,ps::Costs(n,1),i,[&](ps::Bits u,ps::Bits x){
#endif
            auto weight=x.count();
            if(!weight||weight>unsigned(cutoff)||ps::polar(u,n)!=x)throw std::runtime_error("Invalid generated word");
            for(auto&check:checks)if(ps::parity(u&check))throw std::runtime_error("Supercode check violation");
            bool original=true;for(auto&check:removed_checks)if(ps::parity(u&check))original=false;
            if(original){if(weight==unsigned(cutoff))++parent_target;else ++parent_low;}
            if(count++)words<<",";words<<"[\""<<u.to_string()<<"\","<<weight<<"]";
        });
        std::ostringstream out;out<<std::setprecision(17)<<"{\"exact\":true,\"dimension\":"<<prep.transmitted
           <<",\"cutoff\":"<<cutoff<<",\"parent_target\":"<<parent_target<<",\"parent_low\":"<<parent_low
           <<",\"count\":"<<count<<",\"work\":"<<work.total<<",\"seconds\":"<<ps::now()-begin
           <<",\"words\":["<<words.str()<<"]"
#ifdef PS_CAUSAL_INFORMATION_LEADERS
           <<",\"catalogue_scope\":\"all_candidate_words_original_information_leaders\""
#else
           <<",\"catalogue_scope\":\"entire_relaxed_code\""
#endif
           <<"}";result=out.str();
    }catch(const std::exception&e){
        std::string escaped;for(char c:std::string(e.what())){if(c=='"'||c=='\\')escaped+='\\';if(c>=32)escaped+=c;}
        result="{\"exact\":false,\"error\":\""+escaped+"\"}";
    }catch(...){result="{\"exact\":false,\"error\":\"Unknown native failure\"}";}
    char*memory=static_cast<char*>(std::malloc(result.size()+1));if(memory)std::memcpy(memory,result.c_str(),result.size()+1);return memory;
}
