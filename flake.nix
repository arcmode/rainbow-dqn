{
  description = "Development shell using flake-utils";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        ccxt = (
          # modified from https://github.com/nix-community/nur-combined/blob/master/repos/dschrempf/pkgs/finance/ccxt/default.nix#L32
          pkgs.python3Packages.buildPythonPackage rec {
            pname = "ccxt";
            version = "3.0.59";
            format = "pyproject";
            src = pkgs.fetchFromGitHub {
              owner = pname;
              repo = pname;
              rev = "50794fa1b26f3fbb200725c6a6e657454a2cb3a7";
              sha256 = "sha256-MAaQO5q+DmU12r3YOT0cBuwHO68ty2EWgXupdcNo7lM=";
            };
            nativeBuildInputs = with pkgs.python3Packages; [
              setuptools
              wheel
            ];
            prePatch = "cd python";

            # nativeBuildInputs = with python3.pkgs; [ aiodns certifi yarl ];
            propagatedBuildInputs = with pkgs.python3Packages; [
              aiohttp
              cryptography
              requests
            ];

            doCheck = false;
            pythonImportsCheck = [ pname ];
          }
        );
        gym-trading-env = (
          pkgs.python3Packages.buildPythonPackage {
            pname = "gym-trading-env";
            version = "0.3.2";
            format = "pyproject";
            src = pkgs.fetchFromGitHub {
              owner = "sachetz";
              repo = "Gym-Trading-Env";
              rev = "9e91f417c8c0bfafea219a7dda5e08f9ab1c921d";
              sha256 = "sha256-xg/9tGbJXl78u5JEjEjxkTSERy7+/TF4eN44EMhMD5c=";
            };
            nativeBuildInputs = with pkgs.python3Packages; [
              setuptools
              wheel
            ];
            propagatedBuildInputs = with pkgs.python3Packages; [
              pandas
              numpy
              gymnasium
              flask
              pyecharts
              ccxt
              nest-asyncio
              tqdm
              distutils
              scikit-learn
              matplotlib
            ];
          }
        );
      in
      {
        devShell = pkgs.mkShell {
          buildInputs =
            with pkgs;
            with python3Packages;
            [
              python3
              pip
              # jupyter
              jupyterlab
              ipywidgets
              # project dependencies
              torch
              dill
              gym-trading-env
            ];
        };
      }
    );
}
