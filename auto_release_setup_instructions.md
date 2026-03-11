
# Instructions for first release and automatic publishing setup

Follow the steps below to setup automatic releases on PyPI/TestPyPI :

- Sign up to https://test.pypi.org/ and https://pypi.org/ : 
    - You, will need to provide an email adress and use a TOTP app (like MS Authenticator)
    - You will also need to setup 2FA on your account before creating a new package
    - Once this is done, you should create an API token which will be needed for the initial publish
- Once your PyPI/TestPyPI is setup, you can create the project by manually publishing a first version as follows :
    - In the repo, run the command `uv build`, this will generate a wheel and an archive in a `dist` folder
    - run the command `uv publish --token [your API token]` to publish the package on PyPI, to publish on TestPyPi, add the option `--index testpypi`
- After a first publication is made, you can setup automatic releases from the github repo by :
    - Setting up an environment in the github repo settings (admin rights required). You need to setup two environments : pypi and testpypi for each platform.
    - Configure the pypi environment to require manual approval for future releases and add reviewer(s).
    - Navigate into the project page on your personal (Test)PyPI space, section "Publishing"
    - Fill the fields for the Github publisher with Owner = EmertonData , Repository name = glide , Workflow name = release.yml , environment = [pypi or testpypi]

- Add a PyPi badge on the readme linking to the pypi page


Useful Links : 

https://packaging.python.org/en/latest/tutorials/packaging-projects/
https://docs.astral.sh/uv/guides/package/