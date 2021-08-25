---
title: "Compiler Research - Team"
layout: gridlay
excerpt: "Compiler Research: Team members"
sitemap: false
permalink: /team/
---

# Group Members

Jump to [staff](#staff), [students](#students) 

## Staff
{% assign number_printed = 0 %}
{% for member in site.data.team_members %}

{% assign even_odd = number_printed | modulo: 2 %}

{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-6 clearfix">
  <img src="{{ site.url }}{{ site.baseurl }}/images/team/{{ member.photo }}" class="img-responsive" width="25%" style="float: left" />
  <h4>{{ member.name }}</h4>
  <i>{{ member.info }}<br>email: <{{ member.email }}></i>
  <ul style="overflow: hidden">

  {% if member.number_educ == 1 %}
  <li> {{ member.education1 }} </li>
  {% endif %}

  {% if member.number_educ == 2 %}
  <li> {{ member.education1 }} </li>
  <li> {{ member.education2 }} </li>
  {% endif %}

  {% if member.number_educ == 3 %}
  <li> {{ member.education1 }} </li>
  <li> {{ member.education2 }} </li>
  <li> {{ member.education3 }} </li>
  {% endif %}

  {% if member.number_educ == 4 %}
  <li> {{ member.education1 }} </li>
  <li> {{ member.education2 }} </li>
  <li> {{ member.education3 }} </li>
  <li> {{ member.education4 }} </li>
  {% endif %}

  {% if member.number_educ == 5 %}
  <li> {{ member.education1 }} </li>
  <li> {{ member.education2 }} </li>
  <li> {{ member.education3 }} </li>
  <li> {{ member.education4 }} </li>
  <li> {{ member.education5 }} </li>
  {% endif %}

  </ul>
</div>

{% assign number_printed = number_printed | plus: 1 %}

{% if even_odd == 1 %}
</div>
{% endif %}

{% endfor %}

{% assign even_odd = number_printed | modulo: 2 %}
{% if even_odd == 1 %}
</div>
{% endif %}

## Students
{% assign number_printed = 0 %}
{% for member in site.data.students %}
{% if member.current %}
{% assign even_odd = number_printed | modulo: 2 %}

{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-6 clearfix">
  <img src="{{ site.url }}{{ site.baseurl }}/images/team/{{ member.photo }}" class="img-responsive" width="25%" style="float: left" />
  <h4>{{ member.name }}</h4>
  <i>{{ member.info }}<br>email: <{{ member.email }}></i>
  {% unless member.photo == "rock.jpg" %}
  <p> <strong>Education:</strong> {{ member.education }} </p>
  <p class="text-justify">
    <strong>Project description:</strong> {{ member.description }}
  </p>
  <p>
    <strong>Project Proposal:</strong>
    <a href="{{ member.proposal }}" target=_blank >URL</a>
  </p>
  <p> <strong>Mentors:</strong> {{ member.mentors }} </p>
  {% if member.past_projects %}
  <h5>Past Projects</h5>
  <i>{{ member.past_info }}</i>
  <p class="text-justify">
    <strong>Project description:</strong> {{ member.past_description }}
  </p>
  <p>
    <strong>Project Proposal:</strong>
    <a href="{{ member.past_proposal }}" target=_blank >URL</a>
  </p>
  <p> <strong>Mentors:</strong> {{ member.past_mentors }} </p>
  {% assign number_printed = number_printed | plus: 1 %}
  {% endif %}
  {% endunless %}
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

### Former students
{% assign number_printed = 0 %}
{% for member in site.data.students %}
{% unless member.current %}

{% assign even_odd = number_printed | modulo: 2 %}

{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-6 clearfix">
  <h4>{{ member.name }}</h4>
  <i>{{ member.info }}<br>email: <{{ member.email }}></i>
  <p> <strong>Education:</strong> {{ member.education }} </p>
  <p class="text-justify">
    <strong>Project description:</strong> {{ member.description }}
  </p>
  <p> <strong>Final Report:</strong> {{ member.report }} </p>
  <p> <strong>Mentors:</strong> {{ member.mentors }} </p>
  {% if member.past_projects %}
  <h5>Past Projects</h5>
  <i>{{ member.past_info }}</i>
  <p class="text-justify">
    <strong>Project description:</strong> {{ member.past_description }}
  </p>
  <p>
    <strong>Final Report:</strong>
    <a href="{{ member.final_report }}" target=_blank >URL</a>
  </p>
  <p> <strong>Mentors:</strong> {{ member.past_mentors }} </p>
  {% assign number_printed = number_printed | plus: 1 %}
  {% endif %}
</div>

{% assign number_printed = number_printed | plus: 1 %}

{% if even_odd == 1 %}
</div>
{% endif %}
{% endunless %}

{% endfor %}

{% assign even_odd = number_printed | modulo: 2 %}
{% if even_odd == 1 %}
</div>
{% endif %}
