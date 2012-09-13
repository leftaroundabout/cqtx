#ifndef CUDA_PHMSQFN_ACCELLS_TEST
#define CUDA_PHMSQFN_ACCELLS_TEST

extern "C" {

typedef double phmsqfnaccell_fpt;

typedef struct {
  int ncalcs;

  int stt_argsblocksize
    , stt_argsoffset;
  phmsqfnaccell_fpt *stt_args;    // on device

  phmsqfnaccell_fpt *dyn_args;     // on host

  int retsblocksize, retsoffset; phmsqfnaccell_fpt *returns;
}device_concrrcalc_domains;


phmsqfnaccell_fpt *phmsqfnaccell_device_allocate(int n);
void phmsqfnaccell_device_injectdata( int ncalcs, int calcblocksize
                                    , int injoffset, int injcolumns
                                    , phmsqfnaccell_fpt *source[]
                                    , phmsqfnaccell_fpt *destination         );
void phmsqfnaccell_device_retrievedata( int ncalcs, int calcblocksize
                                      , int injoffset, int injcolumns
                                      , phmsqfnaccell_fpt *destination[]
                                      , phmsqfnaccell_fpt *source            );
void phmsqfnaccell_device_freeall(device_concrrcalc_domains domns);


void phmsqfnaccell_polynomial_calc(int degree, device_concrrcalc_domains domns);

}

#endif