R__LOAD_LIBRARY(build/lib_unfold)

void ANfit() {

	Unfold* up = new Unfold();
	up->Fit();
	up->SaveFig("plots/AN_fits.png");
	up->Close();
}