// Exact low-weight catalogue of C + span(polar_row(j)). Vendor stays unchanged.
#include "../vendor/polar-low-weight-spectrum/cpp/src/kernel.cpp"

extern "C" PS_EXPORT char* row_neighborhood(int n,const uint8_t*info,const uint8_t*supports,
                                            int removed,int cutoff,uint64_t limit) {
    std::string result;
    try {
        if(n<4||n>256||(n&(n-1))||removed<0||removed>=n||!info||!supports||info[removed]||cutoff<1||cutoff>n||!limit)
            throw std::invalid_argument("Invalid row-neighborhood input");
        std::vector<ps::Bits> checks;ps::Bits removed_check=ps::unit(removed);int dimension=0;
        for(int j=0;j<n;++j){
            if(info[j]>1)throw std::invalid_argument("Invalid information mask");
            dimension+=info[j];ps::Bits row=ps::unit(j);
            for(int i=0;i<n;++i){
                int b=supports[j*n+i];
                if(b>1||(b&&(info[j]||!info[i]||i>=j)))throw std::invalid_argument("Noncausal support");
                if(b)row.flip(i);
            }
            if(j==removed)removed_check=row;
            else if(!info[j])checks.push_back(row);
        }
        auto begin=ps::now();auto prep=ps::prepare(n,checks,ps::Bits(),ps::Bits());
        if(prep.transmitted!=dimension+1||prep.kernel)throw std::runtime_error("Supercode dimension mismatch");
        ps::Work work(limit);ps::Solver solver(n,prep.checks,cutoff,0,work);
        std::ostringstream words;uint64_t count=0,parent_target=0,parent_low=0;
        for(int i=0;i<n;++i)if(!prep.checks.count(i))solver.generate(0,ps::Costs(n,1),i,[&](ps::Bits u,ps::Bits x){
            auto weight=x.count();
            if(!weight||weight>unsigned(cutoff)||ps::polar(u,n)!=x)throw std::runtime_error("Invalid generated word");
            for(auto&check:checks)if(ps::parity(u&check))throw std::runtime_error("Generated word violates supercode");
            if(!ps::parity(u&removed_check)){if(weight==unsigned(cutoff))++parent_target;else ++parent_low;}
            if(count++)words<<",";words<<"[\""<<u.to_string()<<"\","<<weight<<"]";
        });
        std::ostringstream out;out<<std::setprecision(17)<<"{\"exact\":true,\"dimension\":"<<dimension+1
           <<",\"row\":"<<removed<<",\"cutoff\":"<<cutoff<<",\"parent_target\":"<<parent_target
           <<",\"parent_low\":"<<parent_low<<",\"count\":"<<count<<",\"work\":"<<work.total
           <<",\"seconds\":"<<ps::now()-begin<<",\"words\":["<<words.str()<<"]}";
        result=out.str();
    }catch(const std::exception&e){
        std::string escaped;for(char c:std::string(e.what())){if(c=='"'||c=='\\')escaped+='\\';if(c>=32)escaped+=c;}
        result="{\"exact\":false,\"error\":\""+escaped+"\"}";
    }catch(...){result="{\"exact\":false,\"error\":\"Unknown native failure\"}";}
    char*memory=static_cast<char*>(std::malloc(result.size()+1));if(memory)std::memcpy(memory,result.c_str(),result.size()+1);return memory;
}
