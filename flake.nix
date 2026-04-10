{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        formatter = pkgs.alejandra;
        devShells.default = pkgs.mkShell {
          NIX_ENFORCE_NO_NATIVE = "";
          buildInputs = with pkgs; [
            cmake
            conda
            hwloc
            numactl
            pkg-config
          ];
        };
      }
    );
}
