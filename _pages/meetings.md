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
<div>
  Connection information: {{meeting.connect}} <br />
</div>
<div>
  Agenda:
  <ul>
    {% for item in meeting.agenda %}
    <li> <strong>{{item.title}}</strong> {% if item.speaker %} ({{item.speaker}}) {% endif %} {{item.link | markdownify}}</li>
    {% endfor %}
   </ul>
</div>
</div>
{% endfor %} 
