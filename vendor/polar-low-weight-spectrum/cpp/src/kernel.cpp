// Algorithm 1. Standard C++17, exact nonnegative integer coefficients.
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ps {
double now(){
#ifdef _WIN32
    LARGE_INTEGER tick,frequency;QueryPerformanceCounter(&tick);QueryPerformanceFrequency(&frequency);
    return double(tick.QuadPart)/double(frequency.QuadPart);
#else
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}
#ifndef POLAR_SPECTRUM_CAPACITY
#define POLAR_SPECTRUM_CAPACITY 256
#endif
constexpr int capacity=POLAR_SPECTRUM_CAPACITY;
using Bits=std::bitset<capacity>;
using Basis=std::map<int,Bits>;
struct Big {
    static constexpr uint64_t base=1000000000;
    std::vector<uint32_t> d;
    Big(uint64_t v=0) { while(v){d.push_back(v%base);v/=base;} }
    bool zero() const {return d.empty();}
    Big& operator+=(const Big& b) {
        size_t sz=std::max(d.size(),b.d.size());d.resize(sz);
        uint64_t carry=0;
        for(size_t i=0;i<sz;++i){uint64_t z=uint64_t(d[i])+carry+(i<b.d.size()?b.d[i]:0);d[i]=z%base;carry=z/base;}
        if(carry)d.push_back(carry);return *this;
    }
    friend Big operator*(const Big&a,const Big&b){
        if(a.zero()||b.zero())return Big();
        Big c;c.d.resize(a.d.size()+b.d.size());
        for(size_t i=0;i<a.d.size();++i){
            uint64_t carry=0;
            for(size_t j=0;j<b.d.size();++j){
                uint64_t z=uint64_t(a.d[i])*b.d[j]+c.d[i+j]+carry;
                c.d[i+j]=z%base;carry=z/base;
            }
            c.d[i+b.d.size()]=carry;
        }
        while(!c.d.empty()&&!c.d.back())c.d.pop_back();return c;
    }
    static Big pow2(int n){Big x(1);for(int i=0;i<n;++i){Big y=x;x+=y;}return x;}
    std::string str()const{
        if(zero())return "0";std::ostringstream s;s<<d.back();
        for(int i=int(d.size())-2;i>=0;--i)s<<std::setw(9)<<std::setfill('0')<<d[i];return s.str();
    }
};
using Poly=std::map<int,Big>;
void add(Poly&a,const Poly&b){for(auto&v:b)a[v.first]+=v.second;}
int high(const Bits&r){for(int j=capacity-1;j>=0;--j)if(r[j])return j;return -1;}
Bits unit(int j){Bits r;r.set(j);return r;}
Bits mask(int n){Bits r;for(int j=0;j<n;++j)r.set(j);return r;}
bool parity(const Bits&r){return r.count()%2;}
bool insert(Basis&b,Bits r){
    while(r.any()){int j=high(r);auto p=b.find(j);if(p==b.end()){b[j]=r;return true;}r^=p->second;}return false;
}
Bits reduce(const Basis&b,Bits r){for(auto it=b.rbegin();it!=b.rend();++it)if(r[it->first])r^=it->second;return r;}
Bits polar(Bits v,int n,bool transpose=false){
    for(int step=1;step<n;step*=2)
        for(int k=0;k<n;k+=2*step)for(int j=k;j<k+step;++j)
            if(transpose){if(v[j])v.flip(j+step);}else if(v[j+step])v.flip(j);
    return v;
}
struct Prepared {Basis checks;int original,shortened,transmitted,kernel;};
Prepared prepare(int n,const std::vector<Bits>&rows,Bits p,Bits s){
    Prepared r;for(auto&v:rows)insert(r.checks,v);r.original=n-r.checks.size();
    for(int j=0;j<n;++j)if(s[j])insert(r.checks,polar(unit(j),n,true));
    r.shortened=n-r.checks.size();Basis projected;std::vector<std::pair<int,Bits>> words;
    for(int i=0;i<n;++i)if(!r.checks.count(i)){
        Bits u=unit(i);for(auto&c:r.checks)if(parity(u&c.second))u.flip(c.first);
        words.push_back({i,polar(u,n)});
    }
    Bits kept=mask(n)&~(p|s);std::vector<int> discarded;
    for(auto it=words.rbegin();it!=words.rend();++it)if(!insert(projected,it->second&kept))discarded.push_back(it->first);
    for(int i:discarded)r.checks[i]=unit(i);
    r.transmitted=projected.size();r.kernel=r.shortened-r.transmitted;return r;
}
struct Work{
    uint64_t limit,total=0;std::map<std::string,uint64_t> stats;
    explicit Work(uint64_t limit):limit(limit){}
    void use(const std::string&name,uint64_t count=1){
        if(count>limit-total)throw std::runtime_error("Algorithm 1 work limit exceeded; increase max_work or lower D.");
        total+=count;stats[name]+=count;
    }
};
using Costs=std::vector<int>;
Costs combine(const Costs&q,bool repeat){
    int m=q.size()/2;Costs r(m);for(int j=0;j<m;++j)r[j]=repeat?q[j]+q[m+j]:std::min(q[j],q[m+j]);return r;
}
int lower(Costs q,int first){while(q.size()>1){int m=q.size()/2;q=combine(q,first>=m);first%=m;}return q[0];}
using Equation=std::pair<Bits,bool>;
using Equations=std::map<int,Equation>;
using Rows=std::vector<std::pair<Bits,Bits>>;
struct Item {int cost,index;Bits column;};
struct State {
    Bits syn;int cost;
    bool operator<(const State&b)const {
        if(cost!=b.cost)return cost<b.cost;
        for(int i=capacity-1;i>=0;--i)if(syn[i]!=b.syn[i])return syn[i]<b.syn[i];return false;
    }
};
struct Family {
    int m,base=0,E,rank;bool inconsistent=false;Bits eta,free,target;
    Equations pivots;std::vector<Item> items;Big factor;Work&work;
    Family(Bits a,Bits s,const Costs&q,const Rows&rows,int D,Work&w):m(q.size()/2),work(w){
        work.use("affine_families");std::vector<std::pair<int,int>> positive;
        for(int j=0;j<m;++j){
            int alpha=q[j],beta=q[m+j],delta=s[j]?std::abs(alpha-beta):alpha+beta;
            if(s[j]){base+=std::min(alpha,beta);eta[j]=alpha>beta;}
            if(delta)positive.push_back({delta,j});else free.set(j);
        }
        E=D-base;inconsistent=E<0;Equations residual;
        for(auto&eq:rows){
            Bits row=eq.first;bool rhs=parity(eq.second&a)^parity(row&eta);bool stored=false;
            while((row&free).any()){
                int j=high(row&free);auto old=pivots.find(j);
                if(old==pivots.end()){pivots[j]={row,rhs};stored=true;break;}
                row^=old->second.first;rhs^=old->second.second;
            }
            if(stored)continue;
            while(row.any()){
                int j=high(row);auto old=residual.find(j);
                if(old==residual.end()){residual[j]={row,rhs};stored=true;break;}
                row^=old->second.first;rhs^=old->second.second;
            }
            if(!stored&&rhs)inconsistent=true;
        }
        rank=residual.size();int k=0;for(auto&eq:residual)target[k++]=eq.second.second;
        factor=Big::pow2(int(free.count())-pivots.size());
        for(auto&v:positive){Bits column;int h=0;for(auto&eq:residual)column[h++]=eq.second.first[v.second];items.push_back({v.first,v.second,column});}
        std::sort(items.begin(),items.end(),[](const Item&a,const Item&b){return std::tie(a.cost,a.index)<std::tie(b.cost,b.index);});
    }
    void patterns(const std::function<void(Bits,int)>&emit){
        if(inconsistent)return;
        std::vector<Item> bounded;for(auto&i:items)if(i.cost<=E)bounded.push_back(i);
        std::vector<Basis> suffix(bounded.size()+1);
        for(int i=int(bounded.size())-1;i>=0;--i){suffix[i]=suffix[i+1];insert(suffix[i],bounded[i].column);}
        std::function<void(int,Bits,Bits,int)> visit=[&](int start,Bits syndrome,Bits error,int cost){
            work.use("deviation_nodes");if(reduce(suffix[start],syndrome).any())return;
            if(syndrome.none())emit(error,cost);
            for(int i=start;i<int(bounded.size());++i){
                const auto&v=bounded[i];if(cost+v.cost>E)break;
                visit(i+1,syndrome^v.column,error|unit(v.index),cost+v.cost);
            }
        };visit(0,target,Bits(),0);
    }
    Poly histogram(int policy){
        Poly result;if(inconsistent)return result;int p=items.size(),h=p?std::min(p,E/items[0].cost):0;
        long double qbound=1,choose=1;for(int j=1;j<=h;++j){choose=choose*(p-j+1)/j;qbound+=choose;}
        long double dpbound=(p+1)*static_cast<long double>(E+1)*std::pow(2.L,rank);
        bool dp=policy==2||(policy==0&&dpbound<m*static_cast<long double>(m)*qbound);
        if(dp){
            work.use("dp_families");std::map<State,Big> states;states[{Bits(),0}]=Big(1);
            for(auto&v:items){
                work.use("dp_states",states.size());auto next=states;
                for(auto&x:states)if(x.first.cost+v.cost<=E)next[{x.first.syn^v.column,x.first.cost+v.cost}]+=x.second;
                states=std::move(next);
            }
            for(auto&x:states)if(x.first.syn==target)result[base+x.first.cost]+=factor*x.second;
        }else{
            work.use("enumerated_families");patterns([&](Bits,int cost){result[base+cost]+=factor;});
        }return result;
    }
    std::vector<Bits> directions()const{
        std::vector<Bits> out;
        for(int j=0;j<m;++j)if(free[j]&&!pivots.count(j)){
            Bits v=unit(j);for(auto&eq:pivots)if(parity(eq.second.first&free&v))v.flip(eq.first);out.push_back(v);
        }return out;
    }
    Bits anchor(Bits error)const{
        for(auto&eq:pivots)if(parity(eq.second.first&error)!=eq.second.second)error.flip(eq.first);return error^eta;
    }
};
struct Solver {
    int n,D,policy;Basis checks;Work&work;std::map<std::pair<int,int>,Rows> contexts;
    Solver(int n,const Basis&c,int D,int policy,Work&w):n(n),D(D),policy(policy),checks(c),work(w){}
    const Rows&rows(int offset,int size){
        auto key=std::make_pair(offset,size);auto it=contexts.find(key);if(it!=contexts.end())return it->second;
        int m=size/2;Rows r;
        for(auto&v:checks)if(v.first>=offset+m&&v.first<offset+size)
            r.push_back({polar((v.second>>(offset+m))&mask(m),m,true),(v.second>>offset)&mask(m)});
        return contexts.emplace(key,std::move(r)).first->second;
    }
    Poly multiply(const Poly&a,const Poly&b){
        Poly out;
        for(auto&x:a){if(b.empty()||x.first+b.begin()->first>D)break;for(auto&y:b){
            if(x.first+y.first>D)break;work.use("coefficient_products");out[x.first+y.first]+=x.second*y.second;
        }}return out;
    }
    std::array<Poly,2> wef(const Costs&q,const std::vector<int>&prefix){
        work.use("wef_nodes");
        if(q.size()==1){std::array<Poly,2> out;out[0][0]=Big(1);if(q[0]<=D)out[1][q[0]]=Big(1);return out;}
        Costs evenq,oddq;for(size_t i=0;i<q.size();i+=2){evenq.push_back(q[i]);oddq.push_back(q[i+1]);}
        std::vector<int> e,o;for(size_t i=0;i+1<prefix.size();i+=2){e.push_back(prefix[i]^prefix[i+1]);o.push_back(prefix[i+1]);}
        auto f=wef(evenq,e),g=wef(oddq,o);
        if(prefix.size()%2){int b=prefix.back();return {multiply(f[b],g[0]),multiply(f[1-b],g[1])};}
        auto a=multiply(f[0],g[0]),b=multiply(f[0],g[1]);add(a,multiply(f[1],g[1]));add(b,multiply(f[1],g[0]));return {a,b};
    }
    void generate(int offset,const Costs&q,int first,const std::function<void(Bits,Bits)>&emit){
        work.use("generation_calls");if(lower(q,first)>D)return;
        if(q.size()==1){work.use("generated_pairs");emit(unit(0),unit(0));return;}
        int m=q.size()/2;
        if(first>=m){generate(offset+m,combine(q,true),first-m,[&](Bits b,Bits t){work.use("generated_pairs");emit(b<<m,t|(t<<m));});return;}
        generate(offset,combine(q,false),first,[&](Bits a,Bits s){
            Family f(a,s,q,rows(offset,q.size()),D,work);auto directions=f.directions();
            std::vector<Bits> du,dx;for(auto&v:directions){du.push_back(polar(v,m)<<m);dx.push_back(v|(v<<m));}
            f.patterns([&](Bits error,int){
                Bits t=f.anchor(error),u=a|(polar(t,m)<<m),x=(s^t)|(t<<m);
                // Reflected Gray traversal without a machine-word-sized loop bound.
                std::function<void(int)> gray=[&](int k){
                    if(k<0){work.use("generated_pairs");emit(u,x);return;}
                    gray(k-1);u^=du[k];x^=dx[k];gray(k-1);
                };gray(int(directions.size())-1);
            });
        });
    }
    Poly count(int offset,const Costs&q,int first){
        work.use("count_calls");if(lower(q,first)>D)return {};
        auto it=checks.upper_bound(offset+first);
        if(it==checks.end()||it->first>=offset+int(q.size())){work.use("wef_terminals");return wef(q,std::vector<int>(first,0))[1];}
        int m=q.size()/2;if(first>=m)return count(offset+m,combine(q,true),first-m);
        Poly out;generate(offset,combine(q,false),first,[&](Bits a,Bits s){Family f(a,s,q,rows(offset,q.size()),D,work);add(out,f.histogram(policy));});return out;
    }
};
std::string run(int n,int row_count,const uint8_t*H,int requested,const uint8_t*P,const uint8_t*S,int policy,uint64_t limit){
    if(n<1||n>capacity||(n&(n-1))||requested<0||row_count<0||policy<0||policy>2||!limit||!P||!S||(row_count&&!H))
        throw std::invalid_argument("Invalid kernel input; n must be a power of two at most 256.");
    Bits p,s;for(int j=0;j<n;++j){if(P[j]>1||S[j]>1||(P[j]&&S[j]))throw std::invalid_argument("Invalid rate matching masks.");p[j]=P[j];s[j]=S[j];}
    std::vector<Bits> input;
    for(int i=0;i<row_count;++i){Bits r;for(int j=0;j<n;++j){if(H[i*n+j]>1)throw std::invalid_argument("H must be binary.");r[j]=H[i*n+j];}input.push_back(r);}
    auto start=now();auto prep=prepare(n,input,p,s);
    int D=std::min(requested,n-int(p.count()+s.count()));Costs q(n,1);for(int j=0;j<n;++j)q[j]=p[j]?0:s[j]?D+1:1;
    auto middle=now();Work work(limit);Solver solver(n,prep.checks,D,policy,work);Poly out;out[0]=Big(1);
    for(int i=0;i<n;++i)if(!prep.checks.count(i))add(out,solver.count(0,q,i));
    if(out[0].str()!="1")throw std::runtime_error("Projection invariant failed.");
    auto end=now();auto seconds=[](double a,double b){return b-a;};
    std::ostringstream json;json<<std::setprecision(17)<<"{\"spectrum\":{";bool comma=false;
    for(auto&v:out){if(comma)json<<",";comma=true;json<<"\""<<v.first<<"\":\""<<v.second.str()<<"\"";}
    json<<"},\"exact\":true,\"n\":"<<n<<",\"cutoff\":"<<requested<<",\"evaluated_cutoff\":"<<D
        <<",\"backend\":\"cpp\",\"policy\":\""<<(policy==0?"auto":policy==1?"enumerate":"dp")<<"\""
        <<",\"original_dimension\":"<<prep.original<<",\"shortened_dimension\":"<<prep.shortened
        <<",\"transmitted_dimension\":"<<prep.transmitted<<",\"kernel_dimension\":"<<prep.kernel
        <<",\"preprocessing_seconds\":"<<seconds(start,middle)<<",\"counting_seconds\":"<<seconds(middle,end)
        <<",\"elapsed_seconds\":"<<seconds(start,end)<<",\"stats\":{\"work_units\":"<<work.total;
    for(auto&v:work.stats)json<<",\""<<v.first<<"\":"<<v.second;json<<"}}";return json.str();
}
} // namespace ps
#ifdef _WIN32
#define PS_EXPORT __declspec(dllexport)
#else
#define PS_EXPORT __attribute__((visibility("default")))
#endif
extern "C" PS_EXPORT char* polar_spectrum_compute(int n,int rows,const uint8_t*H,int D,const uint8_t*P,const uint8_t*S,int policy,uint64_t max_work){
    std::string result;
    try{result=ps::run(n,rows,H,D,P,S,policy,max_work);}
    catch(const std::exception&e){
        std::string message=e.what(),escaped;for(char c:message){if(c=='"'||c=='\\')escaped+='\\';if(c>=32)escaped+=c;}
        result="{\"error\":\""+escaped+"\"}";
    }catch(...){result="{\"error\":\"Unknown native failure.\"}";}
    char*memory=static_cast<char*>(std::malloc(result.size()+1));if(memory)std::memcpy(memory,result.c_str(),result.size()+1);return memory;
}
extern "C" PS_EXPORT void polar_spectrum_free(char*p){std::free(p);}
