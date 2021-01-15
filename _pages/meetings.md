---
title: "Project Meetings"
layout: gridlay
excerpt: "Project Meetings"
sitemap: false
permalink: /meetings/
---



# Project Meetings

{% assign sorted_meetings = site.data.meetinglist | sort: "date" | reverse %}

{% for meeting in sorted_meetings %}
<span id={{meeting.label}}>&nbsp;</span>
<div class="well" style="padding-left: 70px; padding-right: 70px">
  <pubtit>{{ meeting.date | date_to_long_string }} at {{meeting.time_cest}} CEST</pubtit>
<div style="text-indent: 20px;">
  Connection information: {{meeting.connect}} <br />
 </div>
<div style="text-indent: 20px;">
  Agenda:
<ul style="margin-top:-10px;">
  {% for item in meeting.agenda %}
  {% if item.link.url %}
  <li> {{item.title}} (<em>{{item.speaker}} </em>): <a href="{{item.link.url}}">{{ item.link.display }}</a> </li>
  {% else %}
  <li> {{item.title}} (<em>{{item.speaker}} </em>): {{ item.link.display }} </li>
  {% endif %}
  {% endfor %}
</ul>

 </div>
 </div>
{% endfor %} 
