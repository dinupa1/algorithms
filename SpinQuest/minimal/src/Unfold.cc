#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TString.h>
#include <TSystem.h>
#include <TStyle.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TGraphErrors.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooAddPdf.h>
#include <RooFitResult.h>
#include <RooPlot.h>
#include <iostream>

#include "Unfold.h"

using std::cout;
using std::endl;
using namespace RooFit;

RecoHist::RecoHist() {

	pi = TMath::Pi();

	TFile* ifile = TFile::Open("unfold.root", "READ");
	sim = (TTree*)ifile->Get("sim");
	save = (TTree*)ifile->Get("save");

	sim->SetBranchAddress("X_true", &X_true);
	sim->SetBranchAddress("X_reco", &X_reco);
	sim->SetBranchAddress("W_true", &W_true);
	sim->SetBranchAddress("W_true1", &W_true1);
	sim->SetBranchAddress("W_reco1", &W_reco1);


	save->SetBranchAddress("X_reco", &X_reco);
	save->SetBranchAddress("W_true", &W_true);

	can = new TCanvas("can", "can", 800, 800);
	hsim = new TH1D("hsim", "; #phi [rad]; counts", 20, -pi, pi);
	hdata = new TH1D("hdata", "; #phi [rad]; counts", 20, -pi, pi);

	legend = new TLegend(0.4,0.2,0.9,0.3);
	legend->SetBorderSize(0);
}

void RecoHist::Plot() {

	for(int ii = 0; ii < sim->GetEntries(); ii++) {
		sim->GetEntry(ii);
		hsim->Fill(X_reco, W_reco1);
	}


	for(int ii = 0; ii < save->GetEntries(); ii++) {
		save->GetEntry(ii);
		hdata->Fill(X_reco, W_true);
	}
}

void RecoHist::SaveFig(TString pname) {

	hdata->SetLineColor(kMagenta+2);
	hdata->SetLineStyle(2);
	hdata->SetLineWidth(2);

	hsim->SetMarkerColor(kOrange+1);
	hsim->SetMarkerStyle(20);

	double xmax = hdata->GetMaximum();
	hdata->SetMaximum(1.5* xmax);
	hdata->SetMinimum(0.);

	hdata->Draw("HIST");
	hsim->Draw("SAME E1");

	legend->AddEntry(hdata, "fake reco. data", "l");
	legend->AddEntry(hsim, "sim reco. events ", "p");
	legend->Draw();
	can->Update();
	can->SaveAs(pname.Data());
}

void RecoHist::Close() {
	delete sim;
	delete save;
	delete can;
	delete hsim;
	delete hdata;
	delete legend;
}

NLLFit::NLLFit():
	pi(TMath::Pi()),
	phi("phi", "phi", -pi, pi),
	weight("weight", "weight", -10., 10.),
	AN("AN", "AN", 0., 0., 1.),
	model("model", "1. + 0.8* AN* sin(1.5707963 - phi)", RooArgList(AN, phi)),
	ds("ds", "ds", RooArgSet(phi, weight), WeightVar(weight)),
	can(nullptr), sim(nullptr), X_true(0), W_true1(0) {

		TFile* ifile = TFile::Open("unfold.root", "READ");
		sim = (TTree*)ifile->Get("sim");

		sim->SetBranchAddress("X_true", &X_true);
		sim->SetBranchAddress("W_true1", &W_true1);

		can = new TCanvas("can", "can", 800, 800);
	}


void NLLFit::Fit() {

	for(int ii = 0; ii < sim->GetEntries(); ii++) {
		sim->GetEntry(ii);
		phi.setVal(X_true);
		weight.setVal(W_true1);
		ds.add(RooArgSet(phi, weight), weight.getVal());
	}

	model.fitTo(ds, SumW2Error(true), PrintLevel(-1));
}

void NLLFit::SaveFig(TString pname) {

	RooPlot *frame = phi.frame(Name("frame"), Title("; #phi; counts"), Bins(20), DataError(RooAbsData::SumW2));

	ds.plotOn(frame);
	model.plotOn(frame);
	model.paramOn(frame, Layout(0.55));

	frame->Draw();
	can->Update();
	can->SaveAs(pname.Data());
}

void NLLFit::Close() {
	delete can;
	delete sim;
}


Unfold::Unfold(): iterations(2), patience(0.001), AN_i(0.){

	graph1 = new TGraphErrors();
	can0 = new TCanvas("can0", "can0", 800, 800);

	gStyle->SetOptStat(0);
}

void Unfold::Fit() {

	for(int ii = 0; ii < iterations; ii++) {

		cout << "---> iteration " << ii << endl;

		gSystem->Exec("python unfold.py");

		cout << "---> plotting unfolded events at reco. level" << endl;

		RecoHist* rh1 = new RecoHist();
		rh1->Plot();
		TString pname = Form("plots/reco_phi_%d.png", ii);
		rh1->SaveFig(pname.Data());
		rh1->Close();

		cout << "---> fitting unfolded events at true level" << endl;

		NLLFit* nllfit = new NLLFit();
		nllfit->Fit();
		TString pname2 = Form("plots/true_phi_%d.png", ii);
		nllfit->SaveFig(pname2.Data());

		double AN_fit = nllfit->AN.getVal();
		double AN_error = nllfit->AN.getError();

		graph1->SetPoint(ii, ii+1, AN_fit);
		graph1->SetPointError(ii, 0., AN_error);

		cout << "---> unfolded AN = " << AN_fit << " +/- " << AN_error << endl;

		if(AN_fit - AN_i < patience){break;}
		AN_i = AN_fit;

		nllfit->Close();
	}
}


void Unfold::SaveFig(TString pname) {

	graph1->SetTitle("; iteration; A_{N}^{fit}");
	graph1->SetMarkerColor(kOrange+1);
    graph1->SetMarkerStyle(20);

    graph1->Draw("APE1");
    can0->Update();
    can0->SaveAs(pname.Data());
}


void Unfold::Close() {
	delete can0;
	delete graph1;
}