# Website setup


## Setup for local development

#### Docker setup

If you use docker, the following line will build and serve the site locally:

```bash
docker run --rm -v "$PWD:/srv/jekyll" -p 4000:4000 -it jekyll/jekyll:3.8 jekyll serve
```

If you want to enable LiveReload (pages automatically reload when jekyll rebuilds after detecting changes), then use this instead:

```bash
docker run --rm -v "$PWD:/srv/jekyll" \
           -p 4000:4000 -p 35729:35729 \
           -it jekyll/jekyll:3.8 \
           jekyll serve --livereload
```

#### Standard setup

Visit [this page](https://jekyllrb.com/docs/installation/) for information about installing Ruby if your current version is too old; the instructions there form the basis for what you see here, and come in variants for all major operating systems.
You should have Ruby 2.4+ for Jekyll. Since versions of macOS before Catalina with 2.3 (and Apple is dropping scripting language from macOS in the future), you may want a newer version even on a mac. You can use rbenv to manage multiple ruby versions. On macOS with homebrew, you'll want:

```bash
brew install rbenv
```

You'll need to run:

```bash
rbenv init
# Prints out instructions
```

and follow the instructions for your current shell. After you've installed rbenv on your system, use:

```bash
rbenv install 2.7.0
```

to get a current version of ruby. Then, inside the main iris-hep website directory, run:

```bash
rbenv local 2.7.0
```

This will run the Ruby you just built whenever you enter this directory. You'll want to install bundler too:

```bash
gem install bundle
```

(You may want to add `--user-install` here if you are not using rbenv. And if
you don't have permission to install, and you are using rbenv, this means you
forgot to set it up with `rbenv init`.)


### Running locally

The site is built with Jekyll, and is easy to run locally if you have Ruby.

To set up a "bundle" (local virtual environment in Python terms):

```bash
bundle install
```

Now, you can use `bundle exec` to run a command in the new environment you just created, such as:

```bash
bundle exec jekyll serve
```

This will incrementally rebuild if anything changes in your directory. Exit with Control-C.



