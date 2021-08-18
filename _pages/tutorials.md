---
title: "Tutorials"
layout: gridlay
excerpt: "Tutorials"
sitemap: false
permalink: /tutorials/
---


# Related Tutorials

<div class="nomarkul">

{% assign sorted_pubs = site.data.tutorialslist | sort: "date" | reverse %}

<div style="padding-left: 40px;">

{% for tutorial in sorted_pubs %}

{% assign author_list = tutorial.author %}
{% assign sep_string = "," %}
{% assign split_auth = author_list | split:sep_string %}

{% if split_auth.size > 4 %}
{% assign author_list = split_auth[0] | append: ", " | append: split_auth[1] |append: ", " |  append: split_auth[2] |append: ", " |  append: split_auth[3] %}
{% assign author_list = author_list | append: ", " | append: " et. al." %}
{% endif %}
{% assign pubinfo = "" %}


<em>{{ author_list }}, </em> <a href="{{ tutorial.url }}">{{ tutorial.title}}</a> {{pubinfo}} ({{tutorial.date}}).
{% if  tutorial.abstract.size  > 7 %}
  * {{tutorial.abstract}}
{% endif %}
{% endfor %}
</div>
</div>
