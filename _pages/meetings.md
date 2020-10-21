---
title: "Project Meetings"
layout: gridlay
excerpt: "Project Meetings"
sitemap: false
permalink: /meetings/
---


# Project Meetings


{% for meeting in site.data.meetings %}

  <b>{{ meeting.date }} at {{meeting.time_cest}} CEST</b>  <br />
  Connection information: {{meeting.connect}} <br />
  Agenda:
  <ul>
  {% for item in meeting.agenda %}
  <li> {{item.title}} (<em>{{item.speaker}} </em>): <a href="{{ item.link.url }}">{{ item.link.display }}</a> </li>
  {% endfor %}
  </ul>
{% endfor %}
