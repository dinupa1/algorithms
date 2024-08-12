#ifndef _UNFOLD__H_
#define _UNFOLD__H_

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TString.h>
#include <TSystem.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TGraphErrors.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooAddPdf.h>
#include <RooFitResult.h>
#include <RooGenericPdf.h>
#include <TLegend.h>
#include <RooPlot.h>
#include <iostream>

class RecoHist {
	double X_reco;
	double X_true;
	double W_true1;
	double W_reco1;
	double W_true;
	double pi;
	TTree* sim;
	TTree* save;
	TCanvas* can;
	TH1D* hsim;
	TH1D* hdata;
	TLegend* legend;
public:
	RecoHist();
	virtual ~RecoHist(){;}
	void Plot();
	void SaveFig(TString pname);
	void Close();
};


class NLLFit {
	double pi;
	double X_true;
	double W_true1;
	TCanvas* can;
	TTree* sim;
public:
	RooRealVar AN;
	RooRealVar phi;
	RooRealVar weight;
	RooGenericPdf model;
	RooDataSet ds;
	NLLFit();
	virtual ~NLLFit(){;}
	void Fit();
	void SaveFig(TString pname);
	void Close();
};

class Unfold {
	double iterations;
	double AN_i;
	double patience;
	TGraphErrors* graph1;
	TCanvas* can0;
public:
	Unfold();
	virtual ~Unfold(){;}
	void Fit();
	void SaveFig(TString pname);
	void Close();
};

#endif /* _UNFOLD__H_ */