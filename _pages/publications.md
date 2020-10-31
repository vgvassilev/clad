---
title: "Publications"
layout: gridlay
excerpt: "Publications"
sitemap: false
permalink: /publications/
---


# Related Publications

<!---
{% assign number_printed = 0 %}
{% for publi in site.data.publist %}

{% assign even_odd = number_printed | modulo: 2 %}
{% if publi.highlight == 1 %}

{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-6 clearfix">
 <div class="well">
  <pubtit>{{ publi.title }}</pubtit>
  <img src="{{ site.url }}{{ site.baseurl }}/images/pubpic/{{ publi.image }}" class="img-responsive" width="33%" style="float: left" />
  <p>{{ publi.description }}</p>
  <p><em>{{ publi.authors }}</em></p>
  <p><strong><a href="{{ publi.link.url }}">{{ publi.link.display }}</a></strong></p>
  <p class="text-danger"><strong> {{ publi.news1 }}</strong></p>
  <p> {{ publi.news2 }}</p>
 </div>
</div>

{% assign number_printed = number_printed | plus: 1 %}

{% if even_odd == 1 %}
</div>
{% endif %}

{% endif %}
{% endfor %}

{% assign even_odd = number_printed | modulo: 2 %}
{% if even_odd == 1 %}
</div>
{% endif %}

<p> &nbsp; </p>


## Full List of publications
--->
<div class="nomarkul">

{% assign sorted_pubs = site.data.publist | sort: "year" | reverse %}

{% for publi in sorted_pubs %}

{% assign author_list = publi.author %}
{% assign sep_string = "," %}
{% assign split_auth = author_list | split:sep_string %}

{% if split_auth.size > 4 %}
{% assign author_list = split_auth[0] | append: ", " | append: split_auth[1] |append: ", " |  append: split_auth[2] |append: ", " |  append: split_auth[3] %}
{% assign author_list = author_list | append: ", " | append: " et. al." %}
{% endif %}
{% assign pubinfo = "" %}
{% if publi.journal.size > 1 %}
{% assign pubinfo = publi.journal | append: " <b> " | append: publi.volume | append: " </b> " | append: publi.pages %}
{% endif %}


<em>{{ author_list }}, </em> <a href="{{ publi.url }}">{{ publi.title}}</a> {{pubinfo}} ({{publi.year}}).
{% if  publi.abstract.size  > 7 %} 
  * {{publi.abstract}}
{% endif %} 
{% endfor %}
</div>