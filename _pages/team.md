---
title: "Compiler Research - Team"
layout: gridlay
excerpt: "Compiler Research: Team members"
sitemap: false
permalink: /team/
---

# Group Members

<div class="clearfix">

{% assign active_contrib = site.data.contributors | where: "active", "1" %}
{% assign past_contrib = site.data.contributors | where_exp: "item", "item.active == nil" %}

{% assign number_printed = 0 %}
{% for member in active_contrib %}
{% assign even_odd = number_printed | modulo: 2 %}

{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-6 clearfix">
  <img src="{{ site.url }}{{ site.baseurl }}/images/team/{{ member.photo }}" class="img-responsive" width="25%" style="float: left" />
  <h4>{{ member.name }}</h4>
  <i>{{ member.info }}<br>email: <{{ member.email }}></i>
  {% if member.photo == "rock.jpg" %}
  </div>
     {% continue %}
  {% endif %}
  <p> <strong>Education:</strong> {{ member.education }} </p>
  {% for project in member.projects %}
  <p class="text-justify">
    <strong> {{ project.status }} project:</strong>
    <i>{{ project.title }}</i><br/>{{ project.description }}
  </p>
  <p>
    <strong>Project Proposal:</strong>
    <a href="{{ project.proposal }}" target=_blank >URL</a>
  </p>
  <p>
    <strong>Project Reports:</strong>
    {{ project.report | markdownify | remove: '<p>' | remove: '</p>' | strip_newlines}}
  </p>
  <p> <strong>Mentors:</strong> {{ project.mentors }} </p> 
  {% endfor %}
</div>

{% assign number_printed = number_printed | plus: 1 %}

{% if even_odd == 1 %}
</div>
{% endif %}

{% endfor %}
</div>

<hr />

### Alumni

<div class="clearfix">


{% assign number_printed = 0 %}
{% for member in past_contrib %}
{% assign even_odd = number_printed | modulo: 2 %}

{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-6 clearfix">
  {% if member.photo %}
  <img src="{{ site.url }}{{ site.baseurl }}/images/team/{{ member.photo }}" class="img-responsive" width="25%" style="float: left" />
  {% endif %}
  <h4>{{ member.name }}</h4>
  <i>{{ member.info }}<br>email: <{{ member.email }}></i>
  <p> <strong>Education:</strong> {{ member.education }} </p>
  {% for project in member.projects %}
  <p class="text-justify">
    <strong> {{ project.status }} project:</strong>
    <i>{{ project.title }}</i><br/>{{ project.description }}
  </p>
  <p>
    <strong>Project Proposal:</strong>
    <a href="{{ project.proposal }}" target=_blank >URL</a>
  </p>
  <p>
    <strong>Project Reports:</strong>
    {{ project.report | markdownify | remove: '<p>' | remove: '</p>' | strip_newlines}}
  </p>
  <p> <strong>Mentors:</strong> {{ project.mentors }} </p> 
  {% endfor %}
</div>

{% assign number_printed = number_printed | plus: 1 %}

{% if even_odd == 1 %}
</div>

{% endif %}

{% endfor %}
</div>
